#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##  Created on: Dec 14, 2018
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import collections
from functools import reduce
import numpy as np
import os
from .mxreason import MXReasoner, ClassEnc
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysat.solvers import Solver as SATSolver
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
from pysmt.shortcuts import Equals, GT, Int, INT, Real, REAL, GE, LE, LT
import resource
from six.moves import range
import sys


#
#==============================================================================
class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb
        
        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))

        # theory
        self.oracle.add_assertion(formula)
        
        if self.optns.attack:
            self.encode_attacker()

        # current selector
        self.selv = None

        # save and use dual explanations whenever needed
        self.dualx = []

        # number of oracle calls involved
        self.calls = 0

    def encode_attacker(self):
        # add last layer of Adversarial Attack classifier
        biased = [r for r in self.xgb.biasLayer] 
        unbiased = [r for r in self.xgb.unbiasLayer]
            
            
        self.adv_outs = [Symbol(f'label_{i}', typename=BOOL) for i in \
                          set({r['class'] for r in (biased+unbiased)})]
        # self.oracle.add_assertion(AtMost(adv_outs,1))
        self.adv_targets = list(set({r['class'] for r in (biased+unbiased)}))
        
            
        # compas, german data: ood => output=0 otherwise =1 (in dist)
        ood_id = self.xgb.target_name.index(0)  
            
        for k,rule in enumerate(biased+unbiased):
            fids = [self.xgb.feature_names.index(f) for f in rule if (f!='class')]
            vals = [rule[f] for f in rule if (f!='class')]
            # find feat id in inps
            for i,j in enumerate(fids):
                if j in self.xgb.categorical_features:
                    vals[i] = self.xgb.encoder[j].transform([[vals[i]]])
                    fids[i] = self.xgb.transform_by_value((self.xgb.feature_names[j],vals[i]))
                else:
                    fids[i] = self.xgb.transform_by_value((self.xgb.feature_names[j],None))
            # encoding rule list into SMT assertions 
            lits = [] 
            if k < len(biased):
                lits.append(GT(self.outs[1-ood_id], self.outs[ood_id]))
            else:
                lits.append(GT(self.outs[ood_id], self.outs[1-ood_id]))
                
            for fid, v in zip(fids, vals):
                if '_' in self.inps[fid].symbol_name():
                    lits.append(self.inps[fid])
                else:
                    operator = None
                    if "<=" in v['operator']:
                        operator = LE
                    elif ">=" in v['operator']:
                        operator = GE
                    elif "<" in v['operator']:
                        operator = LT
                    elif ">" in v['operator']:
                        operator = GT
                    elif "=" in v['operator']:
                        operator = Equals
                    else:
                        assert("invalid operator",v['operator'])

                    value = v['threshold']
                    # lits.append(Equals(self.inps[fid],Real(float(v))))
                    lits.append(operator(self.inps[fid],Real(float(value))))

            q = Symbol(f"label_{rule['class']}", typename=BOOL)        
            self.oracle.add_assertion(Implies(And(lits), q))  
            # print(Implies(And(lits), q)) 
                
        
    def prepare(self, instance):
        """
            Prepare the oracle for computing an explanation.
        """
        
        sample, label = instance

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = [] 
        
        if self.optns.attack :
            outvals = [model.get_py_value(o) for o in self.adv_outs]
            self.out_id = max(zip(outvals, range(len(outvals))))[1] # argmax
            
            for i,o in enumerate(self.adv_outs):
                if i != self.out_id:
                    disj.append(o)
                else:
                    self.oracle.add_assertion(Implies(self.selv, Not(o)))
            # self.oracle.add_assertion(Implies(self.selv, Not(self.adv_outs[self.out_id])))  
        else:    
            for i in range(len(self.outs)):
                if i != self.out_id:
                    disj.append(GT(self.outs[i], self.outs[self.out_id]))
        
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False, label=None):
        """
            Hypotheses minimization.
        """

        # reinitializing the number of used oracle calls
        # 1 because of the initial call checking the entailment
        self.calls = 1

        # adapt the solver to deal with the current sample
        self.prepare((sample,label))

        # saving external explanation to be minimized further
        self.to_consider = [True for h in self.rhypos]

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)
        print("optns.xtype", self.optns.xtype)
        print("optns.xnum", self.optns.xnum)
        if self.optns.xtype == 'abd':
            # abductive explanations => MUS computation and enumeration
            if not smallest and self.optns.xnum == 1:
                expls = [self.compute_minimal_abductive()]
            else:
                expls = self.enumerate_abductive(smallest=smallest)
            print("expls",len(expls))
        else:  # contrastive explanations => MCS enumeration
            if self.optns.use_mhs:
                expls = self.enumerate_contrastive()
            else:
                if not smallest:
                    expls = self.enumerate_minimal_contrastive()
                else:
                    # expls = self.enumerate_smallest_contrastive()
                    expls = self.enumerate_contrastive()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expls = list(map(lambda expl: sorted([self.sel2fid[h] for h in expl]), expls))

        if self.dualx:
            self.dualx = list(map(lambda expl: sorted([self.sel2fid[h] for h in expl]), self.dualx))

        if self.verbose:
            if expls[0] != None:
                for expl in expls:
                    preamble = [self.preamble[i] for i in expl]
                    label = self.adv_targets[self.out_id] if self.optns.attack else \
                                                    self.xgb.target_name[self.out_id]
                    if self.optns.xtype == 'abd':
                        print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                    else:
                        print('  explanation: "IF NOT {0} THEN NOT {1}"'.format(' AND NOT '.join(preamble), label))
                    print('  # hypos left:', len(expl))

            print('  time: {0:.2f}'.format(self.time))

        # here we return the last computed explanation
        return expls

    def compute_minimal_abductive(self):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        # filtering out unnecessary features if external explanation is given
        rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]

        # simple deletion-based linear search
        while i < len(rhypos):
            to_test = rhypos[:i] + rhypos[(i + 1):]

            self.calls += 1
            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                rhypos = to_test

        return rhypos

    def enumerate_minimal_contrastive(self):
        """
            Compute a subset-minimal contrastive explanation.
        """

        def _overapprox():
            model = self.oracle.get_model()

            for sel in self.rhypos:
                if int(model.get_py_value(sel)) > 0:
                    # soft clauses contain positive literals
                    # so if var is true then the clause is satisfied
                    self.ss_assumps.append(sel)
                else:
                    self.setd.append(sel)

        def _compute():
            i = 0
            while i < len(self.setd):
                if self.optns.use_cld:
                    _do_cld_check(self.setd[i:])
                    i = 0

                if self.setd:
                    # it may be empty after the clause D check

                    self.calls += 1
                    self.ss_assumps.append(self.setd[i])
                    if not self.oracle.solve([self.selv] + self.ss_assumps + self.bb_assumps):
                        self.ss_assumps.pop()
                        self.bb_assumps.append(Not(self.setd[i]))

                i += 1

        def _do_cld_check(cld):
            self.cldid += 1
            sel = Symbol('{0}_{1}'.format(self.selv.symbol_name(), self.cldid))
            cld.append(Not(sel))

            # adding clause D
            self.oracle.add_assertion(Or(cld))
            self.ss_assumps.append(sel)

            self.setd = []
            st = self.oracle.solve([self.selv] + self.ss_assumps + self.bb_assumps)

            self.ss_assumps.pop()  # removing clause D assumption
            if st == True:
                model = self.oracle.get_model()

                for l in cld[:-1]:
                    # filtering all satisfied literals
                    if int(model.get_py_value(l)) > 0:
                        self.ss_assumps.append(l)
                    else:
                        self.setd.append(l)
            else:
                # clause D is unsatisfiable => all literals are backbones
                self.bb_assumps.extend([Not(l) for l in cld[:-1]])

            # deactivating clause D
            self.oracle.add_assertion(Not(sel))

        # sets of selectors to work with
        self.cldid = 0
        expls = []

        # detect and block unit-size MCSes immediately
        if self.optns.unit_mcs:
            for i, hypo in enumerate(self.rhypos):
                self.calls += 1
                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    expls.append([hypo])

                    if len(expls) < self.optns.xnum:
                        self.oracle.add_assertion(Or([Not(self.selv), hypo]))
                    else:
                        break

        self.calls += 1
        while self.oracle.solve([self.selv]) and (len(expls) < self.optns.xnum):
            self.ss_assumps, self.bb_assumps, self.setd = [], [], []
            _overapprox()
            _compute()

            expl = [list(f.get_free_variables())[0] for f in self.bb_assumps]
            expls.append(expl)

            if len(expls) >= self.optns.xnum:
                break

            self.oracle.add_assertion(Or([Not(self.selv)] + expl))
            self.calls += 1

        self.calls += self.cldid
        return expls if expls else [None]

    def enumerate_abductive(self, smallest=True):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        expls = []

        # just in case, let's save dual (contrastive) explanations
        self.dualx = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                self.calls += 1
                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])
                    self.dualx.append([self.rhypos[i]])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.dualx.append([self.rhypos[i] for i in to_hit])
                else:
                    if self.verbose > 1:
                        print('expl:', hset)

                    expl = [self.rhypos[i] for i in hset]
                    expls.append(expl)

                    if len(expls) != self.optns.xnum:
                        hitman.block(hset)
                    else:
                        break

        return expls

    def enumerate_smallest_contrastive(self):
        """
            Compute a cardinality-minimal contrastive explanation.
        """

        # result
        expls = []

        # computing unit-size MUSes
        muses = set([])
        for hypo in self.rhypos:
            self.calls += 1
            if not self.oracle.solve([self.selv, hypo]):
                muses.add(hypo)

        # we are going to discard unit-size MUSes from consideration
        rhypos = set(self.rhypos).difference(muses)

        # introducing interer cost literals for rhypos
        costlits = []
        for i, hypo in enumerate(rhypos):
            costlit = Symbol(name='costlit_{0}_{1}'.format(self.selv.symbol_name(), i), typename=INT)
            costlits.append(costlit)

            self.oracle.add_assertion(Ite(hypo, Equals(costlit, Int(0)), Equals(costlit, Int(1))))

        # main loop (linear search unsat-sat)
        i = 0
        while i < len(rhypos) and len(expls) != self.optns.xnum:
            # fresh selector for the current iteration
            sit = Symbol('iter_{0}_{1}'.format(self.selv.symbol_name(), i))

            # adding cardinality constraint
            self.oracle.add_assertion(Implies(sit, LE(Plus(costlits), Int(i))))

            # extracting explanations from MaxSAT models
            while self.oracle.solve([self.selv, sit]):
                self.calls += 1
                model = self.oracle.get_model()

                expl = []
                for hypo in rhypos:
                    if int(model.get_py_value(hypo)) == 0:
                        expl.append(hypo)

                # each MCS contains all unit-size MUSes
                expls.append(list(muses) + expl)

                # either stop or add a blocking clause
                if len(expls) != self.optns.xnum:
                    self.oracle.add_assertion(Implies(self.selv, Or(expl)))
                else:
                    break

            i += 1
            self.calls += 1

        return expls

    def enumerate_contrastive(self, smallest=True):
        """
            Compute a cardinality-minimal contrastive explanation.
        """

        # core extraction is done via calling Z3's internal API
        assert self.optns.solver == 'z3', 'This procedure requires Z3'

        # result
        expls = []

        # just in case, let's save dual (abductive) explanations
        self.dualx = []

        # mapping from hypothesis variables to their indices
        hmap = {h: i for i, h in enumerate(self.rhypos)}

        # mapping from internal Z3 variable into variables of PySMT
        vmap = {self.oracle.converter.convert(v): v for v in self.rhypos}
        vmap[self.oracle.converter.convert(self.selv)] = None

        def _get_core():
            core = self.oracle.z3.unsat_core()
            return sorted(filter(lambda x: x != None, map(lambda x: vmap[x], core)), key=lambda x: int(x.symbol_name()[6:]))

        def _do_trimming(core):
            for i in range(self.optns.trim):
                self.calls += 1
                self.oracle.solve([self.selv] + core)
                new_core = _get_core()
                if len(core) == len(new_core):
                    break
            return new_core

        def _reduce_lin(core):
            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    if not self.oracle.solve([self.selv] + list(to_test)):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True
            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _reduce_qxp(core):
            coex = core[:]
            filt_sz = len(coex) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(coex):
                    to_test = coex[:i] + coex[(i + int(filt_sz)):]
                    self.calls += 1
                    if to_test and not self.oracle.solve([self.selv] + to_test):
                        # assumps are not needed
                        coex = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(coex) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(coex) / 2.0
            return coex

        def _reduce_coex(core):
            if self.optns.reduce == 'lin':
                return _reduce_lin(core)
            else:  # qxp
                return _reduce_qxp(core)

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                self.calls += 1
                if not self.oracle.solve([self.selv, self.rhypos[i]]):
                    hitman.hit([i])
                    self.dualx.append([self.rhypos[i]])
                elif self.optns.unit_mcs:
                    self.calls += 1
                    if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                        # this is a unit-size MCS => block immediately
                        hitman.block([i])
                        expls.append([self.rhypos[i]])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.solve([self.selv] + [self.rhypos[h] for h in list(set(range(len(self.rhypos))).difference(set(hset)))]):
                    to_hit = _get_core()

                    if len(to_hit) > 1 and self.optns.trim:
                        to_hit = _do_trimming(to_hit)

                    if len(to_hit) > 1 and self.optns.reduce != 'none':
                        to_hit = _reduce_coex(to_hit)

                    self.dualx.append(to_hit)
                    to_hit = [hmap[h] for h in to_hit]

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verbose > 1:
                        print('expl:', hset)

                    expl = [self.rhypos[i] for i in hset]
                    expls.append(expl)

                    if len(expls) != self.optns.xnum:
                        hitman.block(hset)
                    else:
                        break

        return expls


#
#==============================================================================
class MXExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()
        self.fcats = []

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb

        # MaxSAT-based oracles
        self.oracles = {}
        if self.optns.encode == 'mxa':
            ortype = 'alien'
        elif self.optns.encode == 'mxe':
            ortype = 'ext'
        else:
            ortype = 'int'
        for clid in range(nof_classes):
            self.oracles[clid] = MXReasoner(formula, clid,
                    solver=self.optns.solver,
                    oracle=ortype,
                    am1=self.optns.am1, exhaust=self.optns.exhaust,
                    minz=self.optns.minz, trim=self.optns.trim)

        # a reference to the current oracle
        self.oracle = None

        # SAT-based predictor
        self.poracle = SATSolver(name='g3')
        for clid in range(nof_classes):
            self.poracle.append_formula(formula[clid].formula)

        # determining which features should go hand in hand
        categories = collections.defaultdict(lambda: [])
        for f in self.xgb.extended_feature_names_as_array_strings:
            if f in self.ivars:
                if '_' in f or len(self.ivars[f]) == 2:
                    categories[f.split('_')[0]].append(self.xgb.mxe.vpos[self.ivars[f][0]])
                else:
                    for v in self.ivars[f]:
                        # this has to be checked and updated
                        categories[f].append(self.xgb.mxe.vpos[abs(v)])

        # these are the result indices of features going together
        self.fcats = [[min(ftups), max(ftups)] for ftups in categories.values()]
        self.fcats_copy = self.fcats[:]

        # all used feature categories
        self.allcats = list(range(len(self.fcats)))

        # variable to original feature index in the sample
        self.v2feat = {}
        for var in self.xgb.mxe.vid2fid:
            feat, ub = self.xgb.mxe.vid2fid[var]
            self.v2feat[var] = int(feat.split('_')[0][1:])

        # number of oracle calls involved
        self.calls = 0

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def delete(self):
        """
            Actual destructor.
        """

        # deleting MaxSAT-based reasoners
        if self.oracles:
            for clid, oracle in self.oracles.items():
                if oracle:
                    oracle.delete()
            self.oracles = {}
        self.oracle = None

        # deleting the SAT-based predictor
        if self.poracle:
            self.poracle.delete()
            self.poracle = None

    def predict(self, sample):
        """
            Run the encoding and determine the corresponding class.
        """

        # translating sample into assumption literals
        self.hypos = self.xgb.mxe.get_literals(sample)

        # variable to the category in use; this differs from
        # v2feat as here we may not have all the features here
        self.v2cat = {}
        for i, cat in enumerate(self.fcats):
            for v in range(cat[0], cat[1] + 1):
                self.v2cat[self.hypos[v]] = i

        # running the solver to propagate the prediction;
        # using solve() instead of propagate() to be able to extract a model
        assert self.poracle.solve(assumptions=self.hypos), 'Formula must be satisfiable!'
        model = self.poracle.get_model()

        # computing all the class scores
        scores = {}
        for clid in range(self.nofcl):
            # computing the value for the current class label
            scores[clid] = 0

            for lit, wght in self.xgb.mxe.enc[clid].leaves:
                if model[abs(lit) - 1] > 0:
                    scores[clid] += wght

        # returning the class corresponding to the max score
        return max(list(scores.items()), key=lambda t: t[1])[0]

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        # first, we need to determine the prediction, according to the model
        self.out_id = self.predict(sample)

        # selecting the right oracle
        self.oracle = self.oracles[self.out_id]

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        # correct class id (corresponds to the maximum computed)
        self.output = self.xgb.target_name[self.out_id]

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} == {1}'.format(f, v))
                else:
                    self.preamble.append(str(v))
            print("preparing")
            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        start_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        if self.optns.encode != 'mxe':
            # dummy call with the full instance to detect all the necessary cores
            self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True)

        print("xnum********************************", self.optns.xnum)
        # calling the actual explanation procedure
        self._explain(sample, smallest=smallest, xtype=self.optns.xtype,
                xnum=self.optns.xnum, unit_mcs=self.optns.unit_mcs,
                reduce_=self.optns.reduce)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        self.used_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem

        if self.verbose:
            print("found expls", len(self.expls))
            for expl in self.expls:
                hyps = list(reduce(lambda x, y: x + self.hypos[y[0]:y[1]+1], [self.fcats[c] for c in expl], []))
                expl = sorted(set(map(lambda v: self.v2feat[v], hyps)))
                preamble = [self.preamble[i] for i in expl]
                label = self.xgb.target_name[self.out_id]

                if self.optns.xtype in ('contrastive', 'con'):
                    preamble = [l.replace('==', '!=') for l in preamble]
                    label = 'NOT {0}'.format(label)

                print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble), label))
                print('  # hypos left:', len(expl))

            print('  calls:', self.calls)
            print('  rtime: {0:.2f}'.format(self.time))

        return self.expls

    def _explain(self, sample, smallest=True, xtype='abd', xnum=1,
            unit_mcs=False, reduce_='none'):
        """
            Compute an explanation.
        """

        if xtype in ('abductive', 'abd'):
            # abductive explanations => MUS computation and enumeration
            if not smallest and xnum == 1:
                self.expls = [self.extract_mus(reduce_=reduce_)]
            else:
                self.mhs_mus_enumeration(xnum, smallest=smallest)
        else:  # contrastive explanations => MCS enumeration
            if xnum == 1:
                self.expls = [self.extract_mcs()]
            else:
                self.mhs_mcs_enumeration(xnum, smallest, reduce_)

    def extract_mus(self, reduce_='lin', start_from=None):
        """
            Compute one abductive explanation.
        """

        def _do_linear(core):
            """
                Do linear search.
            """

            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    # actual binary hypotheses to test
                    if not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        return False
                    to_test.add(a)
                    return True
                else:
                    return True

            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))

        def _do_quickxplain(core):
            """
                Do QuickXplain-like search.
            """

            wset = core[:]
            filt_sz = len(wset) / 2.0
            while filt_sz >= 1:
                i = 0
                while i < len(wset):
                    to_test = wset[:i] + wset[(i + int(filt_sz)):]
                    # actual binary hypotheses to test
                    self.calls += 1
                    if to_test and not self.oracle.get_coex(self._cats2hypos(to_test), early_stop=True):
                        # assumps are not needed
                        wset = to_test
                    else:
                        # assumps are needed => check the next chunk
                        i += int(filt_sz)
                # decreasing size of the set to filter
                filt_sz /= 2.0
                if filt_sz > len(wset) / 2.0:
                    # next size is too large => make it smaller
                    filt_sz = len(wset) / 2.0
            return wset

        self.fcats = self.fcats_copy[:]

        # this is our MUS over-approximation
        if start_from is None:
            assert self.oracle.get_coex(self.hypos, full_instance=True, early_stop=True) == None, 'No prediction'

            # getting the core
            core = self.oracle.get_reason(self.v2cat)
        else:
            core = start_from

        if self.verbose > 2:
            print('core:', core)

        self.calls = 1  # we have already made one call

        if reduce_ == 'qxp':
            expl = _do_quickxplain(core)
        else:  # by default, linear MUS extraction is used
            expl = _do_linear(core)

        return expl

    def mhs_mus_enumeration(self, xnum, smallest=False):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (contrastive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.allcats], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            if self.optns.unit_mcs:
                for c in self.allcats:
                    self.calls += 1
                    if self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                        hitman.hit([c])
                        self.duals.append([c])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                hypos = self._cats2hypos(hset)
                coex = self.oracle.get_coex(hypos, early_stop=True)
                if coex:
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(self.hypos).difference(set(hypos)))

                    for h in removed:
                        if coex[abs(h) - 1] != h:
                            unsatisfied.append(self.v2cat[h])
                        else:
                            hset.append(self.v2cat[h])

                    unsatisfied = list(set(unsatisfied))
                    hset = list(set(hset))

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.oracle.get_coex(self._cats2hypos(hset + [h]), early_stop=True):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)

                    self.duals.append([to_hit])
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def extract_mcs(self,start_from=None):
        """
            Compute one contrastive explanation.
        """
        
        if start_from is None:
            core = self.allcats
        else:
            core = start_from
        
        self.calls = 0

        # detect unit-size MCS immediately
        if self.optns.unit_mcs:
            for i, hypo in enumerate(core):
                self.calls += 1
                if self.oracle.get_coex(self._cats2hypos(core[:i]+core[i+1 :]),  early_stop=True):
                    mcs = [i]
                    return mcs
                
        # compute an over approx        
        # setting preferred polarities
        for clid in self.oracle.oracles:
            self.oracle.oracles[clid].oracle.set_phases(self._cats2hypos(core)) 
                
        model = self.oracle.get_coex([], early_stop=True)
        assert (model)
        
        to_test = []
        
        for j in core:
            for h in self._cats2hypos([j]):
                if model[abs(h)-1]>0:
                    to_test.append(j)
                    break
#             else:
#                 approx.append(j)

        core = [j for j in core if (j not in to_test)]
    
        for i, p in enumerate(core):
            if self.oracle.get_coex(self._cats2hypos(to_test + [core[i]]),  early_stop=True):
                to_test.append(core[i]) 
            self.calls += 1

        mcs  = [j for j in core if (j not in to_test)]
        
#         assert(self.oracle.get_coex(self._cats2hypos(to_test),  early_stop=True))
#         for j in mcs:
#             assert(not self.oracle.get_coex(self._cats2hypos(to_test+[j]),  early_stop=True))        
        
        return mcs

    def mhs_mcs_enumeration(self, xnum, smallest=False, reduce_='none', unit_mcs=False):
        """
            Enumerate subset- and cardinality-minimal contrastive explanations.
        """

        # result
        self.expls = []

        # just in case, let's save dual (abductive) explanations
        self.duals = []

        with Hitman(bootstrap_with=[self.allcats], htype='sorted' if smallest else 'lbx') as hitman:
            # computing unit-size MUSes
            for c in self.allcats:
                self.calls += 1

                if not self.oracle.get_coex(self._cats2hypos([c]), early_stop=True):
                    hitman.hit([c])
                    self.duals.append([c])
                elif unit_mcs and self.oracle.get_coex(self._cats2hypos(self.allcats[:c] + self.allcats[(c + 1):]), early_stop=True):
                    # this is a unit-size MCS => block immediately
                    self.calls += 1
                    hitman.block([c])
                    self.expls.append([c])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 2:
                    print('iter:', iters)
                    print('cand:', hset)

                if hset == None:
                    break

                self.calls += 1
                if not self.oracle.get_coex(self._cats2hypos(set(self.allcats).difference(set(hset))), early_stop=True):
                    to_hit = self.oracle.get_reason(self.v2cat)

                    if len(to_hit) > 1 and reduce_ != 'none':
                        to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    self.duals.append(to_hit)

                    if self.verbose > 2:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    if self.verbose > 2:
                        print('expl:', hset)

                    self.expls.append(hset)

                    if len(self.expls) != xnum:
                        hitman.block(hset)
                    else:
                        break

    def _cats2hypos(self, scats):
        """
            Translate selected categories into propositional hypotheses.
        """

        return list(reduce(lambda x, y: x + self.hypos[y[0] : y[1] + 1],
            [self.fcats[c] for c in scats], []))

    def _hypos2cats(self, hypos):
        """
            Translate propositional hypotheses into a list of categories.
        """

        pass
