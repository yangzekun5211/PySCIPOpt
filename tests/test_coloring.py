try:
    import networkx
    import sys
    from networkx.algorithms.approximation import max_clique
    from pyscipopt import Model, Variable, Branchrule, Conshdlr, Pricer, SCIP_RESULT, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    import inspect
    entering = lambda: print("IN ", inspect.stack()[1][3]) # debugging magic
    leaving = lambda: print("OUT ", inspect.stack()[1][3]) # debugging magic
except:
    import pytest
    pytest.skip()
# I guess this is the idea of everything:
# The formulation is:  min{\sum_{s \in S} x_s :  \sum_{s \in S: u \in s} x_s >= 1 for all u \in V}
# where G = (V,E) is the graph we want to color, S is the set of *all* maximal stable sets (i.e. independent sets)
# (i.e. subsets of node of V which are not neighbors) (BTW, N(u) will denote the neighborhood of u)
# x_s is a binary variable and x_s = 1 means that all u \in s receive the same color and x_s = 0 means nothing.
# NOTE 0: if x_s = 1 and x_s' = 1, then they actually receive different colors, hence minimizing min{\sum_{s \in S} x_s
#         minimizes the number of colors. If a node is associated to two colors, I guess one can choose any (?)

# Of course S is huge, so column generation.
# Given the current master with stable sets S':
# - if it is infeasible (pricerfarkas): for each node u such that u \notin s \forall s \in S' (this is the only way
#   for the LP to be infeasible), we find a maximal stable set in G which containt u and add it to S'
# - if it is feasible (the pricing problem) (pricerredcost): given \pi, the dual solution, we solve
#   \max{ \pi s : s is a stable set of G } and if it is larger than 1, we add s to S'
# NOTE 1: It seems that networkx doesn't provide functionality to solve the maximum weight independent set
#         so we handle this by finding *all* independent sets and then dot product each of them with \pi and
#         select the maximum :-) (by them it is meant their indicator vector)
#
# At some point we will have to branch. Now, given a fractional solution of master, consider s \in S' such that
# x_s is fractional. Its fractionality (i.e. x_s < 1) and feasibility implies that for every u \in s, there is
# s' (sorry for the s' and S') such that u \in s' (and x_s' > 0). Also, sin both are maximal, then one *cannot*
# be a subset of the other, which implies that there is w \in V such that w is in s or s', *but not* in both.
# Hence, branching on "u and w have the same color" and "u and w have different colors" will:
# - remove the current LP solution: Suppose s is the one that contains u and w (then s' contains only u). In the
#   branch where SAME(u,w) (this means what it is suppose to mean), x_s' *has* to be 0 (and a lot of other guys, but
#   that is beside the point).
#   In the branch DIFF(u,w), now x_s *has* to be 0 (also a lot of other guys, but still beside the point)
#   NOTE: this other guys thing is imposed in the propagation of the constraints which enforces the SAME and DIFF
# - not remove any feasible solution: there are two types of coloring in the world, the ones in which u and w have
#   the same color, and the ones in which u and w have different color
# NOTE 2: since u and w are in the same stable set (s or s'), then they are *not* neighbors. So imposing that they
#         are different doesn't produce an infeasible subproblem (not sure why this note is important)
# NOTE 3: the beauty of the branching rule is that both subproblems are of the same kind of problem as the original one,
#         just on a different graph
#
# Enforcing DIFF(u,w) amounts to add an edge between u and w, while enforcing
# SAME(u,w) amounts to merge node u and w (i.e. create new node {u,w} (yes, the name is very unfortunate) and if
# e = {v,u} or e = {v,w} existed in the original graph, then e = {v, {u,w}} exists in the "merged" graph)
# The devil is in the details:
# - Every node in the tree will have its own graph.
# - Enforcing DIFF(u,w), in practice, means adding the edge *and* setting to 0 all x_s such that s contains both
#   u and w (since such an x_s being equal to 1, means both variable have the same color; see NOTE 0).
#   The edge will prevent the pricing problem to generate stable sets which contains both u and w
# - Enforcing SAME(u,w), in practice, is a mess. One idea, as done in the C code, is to add enough edges so that
#   N(u) = N(w). This is enough to ensure that every maximal stable set will either, have both, u and w or none of
#   them. [A small proof: let s be a maximal stable set, if u \notin s, then there is v \in N(u) such that v \in s
#   (otherwise, u could be added to s contradicting its maximality). Since v \in N(u) = N(w), then w cannot be in s]
#   We use a different (hopefuly, more elegant; though I guess less efficient... but did we talk already about the
#   brute force way of finding maximum weight independent sets?) approach here. We construct a "quotient graph", i.e.,
#   a graph whose nodes are equivalence classes. The equivalence classes are the subsets of nodes which are equal (yes,
#   we have been talking about enforcing that only u and w have to be equal, but imagine yourself now deep in the tree).
#   NOTE: We only use the quotient graph to search for stable sets. Once a stable set in the quotient graph is generated,
#         we have to map it to a stable set of the original graph (where actually the constraints live), but this simple.
#   Appart from that, we also have to set to 0 all x_s such that s contains u or w but not both.
# NOTE 4: the operations that enforce SAME and DIFF *preserve* maximal stable sets. So if we only add variables associated
#         to maximal stable sets, then in every node of the tree the (valid) stable sets are going to be maximal
#
# More details:
# - We use a constraint handler to store the graph of each node. Given the graph of the parent node, generating the graphs
#   of both childs, DIFF and SAME is done as follows:
#   - DIFF(u,w): just add the corresponding edgea e={u,w}
#   - SAME(u,w): build a new quotient graph, from the original graph. Let G be the original graph and H the parent's graph.
#     Note that H is already a quotient graph of G. The equivalence relation used over the nodes to generate the child's graph
#     is: node1 R node2 if and only if (node1 == u and node2 == w) or [node1]_H == [node2]_H
#     where [v]_H is the equivalence class of v in H ([node1]_H == [node2]_H tests whether node1 and node2 are the same node in H)
# - The graph that needs to be colored is given in some file. This graph gets pre-processed (see the C file for documentation)
#   The "original graph" is the pre-processed graph.
# END

#FIXME delete
#transmodel = 0

##### THIS WOULD BE THE EQUIVALENT OF PROBDATA
class Coloring(Model):
    def __init__(self, graph = None, from_model = None):
        entering()
        if from_model is None:
            super(Coloring, self).__init__() # creates the SCIP: maybe there should be a way to pass the plugins one wants to include
            print("scip is inited, continue")

            # here we take care of the data
            self.originalgraph = graph
            self.graph = graph.copy()
            self.preprocess_graph()
            self.vars = []
        else:
            super(Coloring, self).__init__(from_model=from_model)
        leaving()

    def probtrans(self):
        entering()
        # create "new" model FIXME: correct way of doing this? had to alter the init method!
        print("original model received with id: ", id(self))
        print(self)
        targetmodel = Coloring(from_model=self)
        print("transformed model's id is ", id(targetmodel))

        # copy data
        print("copying data to transformed model")
        targetmodel.originalgraph = self.originalgraph.copy()
        targetmodel.graph = self.graph.copy()
        print("graphs done")
        targetmodel.constraints = {}
        for v in self.graph.nodes_iter():
            targetmodel.constraints[v] = self.getTransformedCons(self.constraints[v])
        print("trans conss done")
        targetmodel.vars = []
        for var in self.vars:
            targetmodel.vars.append(self.transformVar(var))
        print("trans vars done")

        # copy handler!!!!
        targetmodel.conshdlr = self.conshdlr
        #FIXME delete
        #transmodel = targetmodel
        leaving()
        return {"targetdata" : targetmodel}

    def preprocess_graph(self):
        entering()
        print("preprocessing graph: computing max clique")

        # compute maximum clique
        max_clique = max(networkx.find_cliques(self.graph))
        #print("Maximum clique has size ",networkx.graph_clique_number(self.graph))
        #print(max_clique)
        #networkx.draw(self.graph)
        #plt.show()
        #print(self.graph.nodes())

        improvement = True
        while improvement:
            improvement = False
            nnodes_start = self.graph.number_of_nodes()

            # iteration to remove nodes with degree < |max_clique|
            for n in self.graph.nodes():
                #print("degree of node %d is %d"%(n, self.graph.degree(n)))
                if n not in max_clique and self.graph.degree(n) < len(max_clique):
                    self.graph.remove_node(n)
            #print(self.graph.nodes())
            #networkx.draw(self.graph)
            #plt.show()

            # iteration to remove nodes with dominated neighborhoods
            for u in self.graph.nodes():
                u_neighbors = set(self.graph.neighbors(u))
                for v in self.graph.nodes_iter():
                    if u == v or self.graph.has_edge(u,v):
                        continue
                    v_neighbors = set(self.graph.neighbors(v))
                    if u_neighbors.issubset(v_neighbors):
                        self.graph.remove_node(u)
                        break
            #print(self.graph.nodes())
            #networkx.draw(self.graph)
            #plt.show()

            if nnodes_start != self.graph.number_of_nodes():
                improvement = True
        leaving()

    def set_up_constraints(self):
        entering()
        # add empty set covering constraints; this is a dictionary mapping nodes to constraints
        constraints = {}
        for n in self.graph.nodes_iter():
            constraints[n] = self.addCons({}, name="Node-Constraint"+str(n), lhs=1, modifiable = True, dynamic=True)
        # store them in probdata
        self.constraints = constraints
        leaving()
########### END THE PROBDATA


############ THE STABLE SETS ARE THE VARIABLES
class StableSetVar(Variable):
    def __init__(self, scip, stable_set):
        self.stable_set = stable_set
        self.scip = scip #this is model
    def __del__(self): # FIXME
        self.scip.releaseVar(self)

########### END THE VARIABLES

############ THE BRANCHING RULE
class ColoringBranch(Branchrule):
    # local functions (?)
    def find_nodes_to_branch(self, scip, s1):
        entering()
        print("looking for nodes (of graph) to branch on var: ", dir(s1))
        print("stable set ", s1.stable_set)
        for node1 in s1.stable_set: # FIXME: wouldn't we actually need any node???
            print("taking node1  ", node1)
            # FIXME: ideally, we would get the constraint of the node and iterate over its variables (this is
            #        the same as iterating over the stable sets which contain s1), but there is this problem on how to
            #        get constraints from scip
            for s2 in (var for var in scip.vars if node1 in var.stable_set and var.isInLP() and var.getUbLocal() > 0.5):
                print("var with stable set shares node1  ", s2.stable_set)
                # TODO: have to ask if var is active and what not
                for node2 in s2.stable_set:
                    if node2 not in s1.stable_set:
                        leaving()
                        return node1, node2
            #for var in scip.vars:
            #    print("other vars stable set ", var.stable_set)
            #    if node1 in var.stable_set:
            #        print("both vars share the node")

    def branchexeclp(self, allowaddcons):
        entering()

        lpcands, fracvals = self.model.getLPBranchCands()
        assert len(lpcands) > 0
        assert len(self.model.vars) > 0

        # get least fractional candidate
        fractionalities = list(map(lambda x : min(x,1-x), fracvals))
        bestcandidate = fractionalities.index(min(fractionalities))

        # get variable with least fractionality
        s1 = lpcands[bestcandidate]

        # get node1 and node2 and variable s2 such that:
        # - node1 is in the stable sets of s1 and s2
        # - node2 is in *exactly* one of the stable sets
        # FIXME: we mentioned on top that x_s' (x_s2 here) was > 0, but here we don't check this...
        node1, node2 = self.find_nodes_to_branch(self.model, s1)
        print(">>>>>>>>>>>>>>.. found nodes to branch on! ", node1, node2)

        # get current graph: Python's black magic
        current_graph = getCurrentGraph(self.model)

        # assert that nodes are not connected
        assert not current_graph.has_edge(get_node_class(current_graph, node1), get_node_class(current_graph, node1))

        # create children
        childsame = self.model.createChild(0.0, self.model.getLocalTransEstimate())
        childdiff = self.model.createChild(0.0, self.model.getLocalTransEstimate())
        print("children created!")

        # create constraints: Python's black magic
        conssame = self.model.conshdlr.createCons("some_same_name", node1, node2, current_graph, "same", childsame)
        consdiff = self.model.conshdlr.createCons("some_diff_name", node1, node2, current_graph, "diff", childdiff)
        print("constraints created!")

        # add them: maybe Robert will prefer writing in a single line the create and add?
        self.model.addPyConsNode(childsame, conssame)
        self.model.addPyConsNode(childdiff, consdiff)
        print("constraints added!")

        leaving()
        return {"result" : SCIP_RESULT.BRANCHED}

    def branchexecps(self, allowaddcons):
        entering()
        print("I HOPE I DON'T HAVE TO IMPLEMENT THIS GUY!!!")
        leaving()
########### END BRANCHING

############ THE CONSHDLR FOR STOREGRAPH CONSTRAINTS
def get_node_class(H, node1):
    for n in H.nodes_iter():
        if node1 in n:
            return n

def same_node(H, node1, node2):
    n = get_node_class(H, node1)
    return node2 in n

def getCurrentGraph(scip):
    conshdlr = scip.conshdlr #this is the equivalent of scipfindconshdlr
    assert len(conshdlr.stack) > 0
    graph = conshdlr.stack[-1].data["current_graph"]
    assert graph != None
    return graph


class StoreGraphConshdlr(Conshdlr):
    def __init__(self): # here comes the conshdlr data
        entering()
        self.stack = []
        leaving()

    # we do it this way in name of efficieny :) we could just create the graph right away and
    # wouldn't have to bother storing the parent's graph (we need node1 and node2, see prop)
    def createCons(self, name, node1, node2, parent_graph, type, stickingnode):
        entering()
        cons = self.model.createCons(self, name, stickingatnode=True) #FIXME: this is very nasty
        cons.data = {} # don't like this....
        cons.data["node1"] = node1
        cons.data["node2"] = node2
        cons.data["parent_graph"] = parent_graph
        cons.data["type"] = type
        cons.data["stickingatnode"] = stickingnode
        cons.data["npropagatedvars"] = 0
        cons.data["created"] = False
        cons.data["current_graph"] = None
        leaving()
        return cons

    # initsol: branch and bound is going to start now, so we create the constraint containing the graph of the root node
    def consinitsol(self, constraints):
        entering()
        print("received conshdlr: ",id(self))
        print("received model: ",id(self.model))
        #print("received scip: ",id(scip))
        #print("transmodel : ",id(transmodel))
        #assert transmodel == scip
        cons = self.createCons("root", -1,-1, None, "root", None)
        cons.data["created"] = True
        cons.data["current_graph"] = networkx.quotient_graph(self.model.graph, lambda u,v: u == v)
        self.stack.append(cons)
        assert len(self.stack) == 1
        leaving()

    # consactive: [more info in FAQ of SCIP]
    # In general, we are adding constraints to nodes. Every time the node is entered, the consactive callback
    # of the constraints added to the node are called. So here we do
    # - if entered for the first time, create the data of the constraint (graph, etc)
    # - if re-entered, check if new vars were generated in between calls. We have to repropagate the node in the affirmative case
    # - place the constraint on top of the stack (to know which constraint is the current active constraint, i.e. to know the
    #   current graph)
    def consactive(self, constraint):
        entering()
        self.stack.append(constraint)

        consdata = constraint.data
        if not consdata["created"]:
            type         = consdata["type"]
            node1        = consdata["node1"]
            node2        = consdata["node2"]
            parent_graph = consdata["parent_graph"]
            print("creating consdata for cons: ", constraint)
            print("node1 and node2: ", node1, node2)
            print("parent graphs nodes ", parent_graph.nodes())
            print("parent graphs edges ", parent_graph.edges())

            if type == "same":
                consdata["current_graph"] = networkx.quotient_graph(self.model.graph,
                        lambda u,v: (u == node1 and v == node2) or same_node(parent_graph, u, v))
            elif type == "diff":
                node1_class = get_node_class(parent_graph, node1)
                node2_class = get_node_class(parent_graph, node2)
                print("nodeclass1 ", node1_class)
                print("nodeclass2 ", node2_class)
                assert node1_class != node2_class
                consdata["current_graph"] = parent_graph.copy()
                consdata["current_graph"].add_edge(node1_class, node2_class)
            else:
                raise ValueError("type %s unkonwn"%type)
        elif consdata["npropagatedvars"] < self.model.getNTotalVars() and consdata["type"] != "root":
            self.model.repropagateNode(constraint.data["stickingatnode"])

        consdata["created"] = True
        leaving()

    # here we just remove the constraint from the stack
    def consdeactive(self, constraint):
        assert constraint == self.stack.pop() # I hope this compares the ids

    # propagation: we have to set to 0 all variable associated to invalid stable sets
    # we *only* need to check whether the stable sets contain node1 and node2 when they shouldn't
    # or don't contain both when they should (i.e. we do not have to check whether the stable set
    # is valid in the quotient graph, because all unfixed variables represent stable sets which are
    # valid for the parent graph and the only difference between the parent and ours is SAME(node1,node2)
    # or DIFF(node1, node2))
    def consprop(self, constraints, nusefulconss, nmarkedconss, proptiming):
        entering()
        print("selfs scip has nvars = ", len(self.model.vars))
        # get the only constraint we care about
        consdata = self.stack[-1].data
        if consdata["type"] == "diff":
            for var in self.model.vars:
                if var.isInLP() and var.getUbLocal() > 0.5:
                    if consdata["node1"] in var.stable_set and consdata["node2"] in var.stable_set:
                        self.model.chgVarUb(var, 0.0)
        if consdata["type"] == "same":
            for var in self.model.vars:
                if var.isInLP() and var.getUbLocal() > 0.5:
                    if len(var.stable_set.intersection(set([consdata["node1"], consdata["node2"]]))) == 1:
                        self.model.chgVarUb(var, 0.0)
        consdata["npropagatedvars"] = len(self.model.vars)
        leaving()
        return {"result": SCIP_RESULT.DIDNOTFIND}

    # fundamental callbacks do nothing
    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        entering()
        leaving()
        return {"result": SCIP_RESULT.FEASIBLE}
    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        entering()
        leaving()
        return {"result": SCIP_RESULT.FEASIBLE}
    def conslock(self, constraint, nlockspos, nlocksneg):
        entering()
        leaving()
########### END THE CONSHDLR

############ THE PRICER
class ColoringPricer(Pricer):

    # set up some basic data
    def __init__(self):
        entering()
        self.maxvarsround = 2     # maximal number of vars created per round UNUSED
        self.nstablesetsfound = 0 # number of improving stable sets found
        self.nround = 0
        self.max_rounds = 10 # how to set parameter?
        self.current_graph = None
        leaving()

    # farkas pricing method of variable pricer for infeasible LPs
    def pricerfarkas(self):
        entering()
        # get current node's graph
        print("getting stored graph")
        G = getCurrentGraph(self.model)

        # get all stable sets (all vars)
        vars = self.model.vars

        # mark all nodes in the current stable sets as colored: colored_nodes = union_{s \in stable_sets} s
        colored_nodes = set()
        for var in vars:
            if var.isInLP() and var.getUbLocal() > 0.5: # FIXME: this is different from the c example!
                colored_nodes.update(var.stable_set)

        print("looking for max st sets")
        # build maximal stable sets until all nodes are colored
        uncolored_nodes = set(self.model.graph.nodes()).difference(colored_nodes)
        while len(uncolored_nodes) > 0:
            # build maximal stable set
            #print("nodes ",G.nodes())
            #print("edges ",G.edges())
            node_class = get_node_class(G, uncolored_nodes.pop())
            #print("node class", node_class)
            quotients_stable_set = networkx.maximal_independent_set(G, [node_class])
            new_stable_set = set()
            for node_set in quotients_stable_set:
                new_stable_set.update(node_set)

            # update uncolored nodes
            uncolored_nodes -= new_stable_set

            # create var associated with the new stable set
            # FIXME: - check that it is actually new
            #        - the c code marks variable as deletable and has an event handler
            #          for when a variable is deleted... we don't do any of that here
            print("found max stable set, creating var now")
            newVar = StableSetVar(self.model, new_stable_set)
            self.model.addPyVar(newVar, name="StableVar"+str(len(vars)), vtype="B", obj = 1, pricedVar = True)
            print("done creating var appending")
            vars.append(newVar)
            print("done appending adding PyVar!")
            print("done adding PyVar!")
            self.model.chgVarUb(newVar, 1.0, lazy=True)

            # add variable to constraints
            for v in new_stable_set:
                self.model.addConsCoeff(self.model.constraints[v], newVar, 1.0)
        print("all nodes are colored!")
        leaving()
        return {'result':SCIP_RESULT.SUCCESS}

    # FIXME: make me more like c (?)
    def keep_on_pricing(self):
        print("should I keep on pricing? round ", self.nround)
        if self.current_graph == getCurrentGraph(self.model):
            self.nround += 1
            return self.nround < self.max_rounds
        else:
            self.nround = 0
            self.lowerbound = -self.model.infinity()
            self.current_graph = getCurrentGraph(self.model)
            return True

    # The reduced cost function for the variable pricer
    def pricerredcost(self):
        entering()

        # stop pricing if limit for pricing rounds reached
        if not self.keep_on_pricing():
            print("maxrounds reached, pricing interrupted")
            leaving()
            return {'lowerbound' : self.lowerbound, 'result' : SCIP_RESULT.DIDNOTRUN }

        # get dual solution
        pi = {}
        for n in self.model.graph.nodes_iter():
            pi[n] = self.model.getDualsolLinear(self.model.constraints[n]) #TODO: should we receive model instead of storing it in Pricer?

        # get current graph
        G = getCurrentGraph(self.model)

        # brute force (see NOTE 1) cliques of the complement are stable sets of the original
        max_stable_value = 0
        max_stable_set = []
        for quotients_stable_set in networkx.enumerate_all_cliques(networkx.complement(G)):
            # transform quotient's stable set to original's stable set
            stable_set = set()
            for node_set in quotients_stable_set:
                stable_set.update(node_set)

            # compute value
            stable_value = 0 # what is the pythonic way of doing this?
            for n in stable_set:
                stable_value += pi[n]

            if stable_value > max_stable_value:
                max_stable_value = stable_value
                max_stable_set = stable_set

        assert(len(max_stable_set) > 0)
        assert(max_stable_value > 0)

        print("round %d, max stable set value found: %g"%(self.nround,max_stable_value))
        # add maximum only if not already inside
        if max_stable_value > 1 + 1e-6: #FIXME
            self.model.writeProblem(filename="coloring_round_%s.cip"%(str(self.nround)), original=False)
            if max_stable_set not in list(var.stable_set for var in self.model.vars):
                # create var associated with the max stable set FIXME: code repetition
                newVar = StableSetVar(self.model, max_stable_set)
                self.model.addPyVar(newVar, name="StableVar"+str(len(self.model.vars)), vtype="B", obj = 1, pricedVar = True)
                self.model.vars.append(newVar)
                self.model.chgVarUb(newVar, 1.0, lazy=True)

                # add variable to constraints
                for v in max_stable_set:
                    self.model.addConsCoeff(self.model.constraints[v], newVar, 1.0)
            else:
                print("stable set found is already present on stable sets redcost: ", max_stable_value)
                print("stable set found ", max_stable_set)
                #raise ValueError("SHOULD THIS HAPPEN?")

            if self.model.isLPOptimal():
                self.lowerbound = max(self.lowerbound, self.model.getLPObjval()+(1.0 - max_stable_value)*self.model.getPrimalbound())
        leaving()
        return {'result' : SCIP_RESULT.SUCCESS }
############# END THE PRICER

# FIXME: this is the job of the reader I thought it didn't make sense to have a reader
# for this, but I forgot why....
def create_problem():
    entering()
    print("reading file ", sys.argv[-1])
    # open file
    f = open(sys.argv[-1], 'r')
    # if fails: return {"result": SCIP_RESULT.READERROR}
    # get nodes of graph
    for line in f:
        if "p" == line[0]:
            nedges = int(line.split()[2])
            break
    G = networkx.Graph()
    G.add_nodes_from(range(1,nedges+1))
    # get edges of graph
    for line in f:
        if "e" == line[0]:
            s = line.split()
            G.add_edge(int(s[1]), int(s[2]))
    print(G.nodes())
    print(G.edges())

    print("CREATING THE ORIGINAL COLORING OBJECT NOW")
    color_scip = Coloring(G)
    print("object created id: ", id(color_scip))
    color_scip.printVersion() # just to check that scip is not null!

    # set up constraints: a bunch of empty setcover constraints (sum x_i >= 1)
    print("create LP")
    color_scip.set_up_constraints()

    # tell scip objective is integral
    color_scip.setObjIntegral()

    leaving()
    return color_scip
    # get problem name (can't get them, don't understand python's problem with strings)
    #name = file.rsplit('/',1)[-1][:-3]
    #print("problem name is ", name)

def test_main():
    entering()

    print("creating problem")
    scip = create_problem()

    print("problem setted: printing it")
    scip.writeProblem("setup_coloring_orig.cip")
    scip.printVersion()

    # include branching rule
    print("including branchingrule")
    branching_rule = ColoringBranch()
    scip.includeBranchrule(branching_rule, "coloring", "branching rule for coloring", 50000, -1, 1.0)
    print("done including branchingrule: id ", id(branching_rule))

    # include conshdlr
    print("including conshdlr")
    conshdlr = StoreGraphConshdlr()
    scip.includeConshdlr(conshdlr, "storeGraph", "storing graph at nodes of the tree constraint handler",
                         chckpriority=2000000, propfreq=1)
    scip.conshdlr = conshdlr # we store directly to simulate the scipfindconshdlr though I guess the correct
                             # thing to do is to implement scipfidconshdlr ?? FIXME
    print("done including conshdlr: id ", id(conshdlr))
    print("conshdlr is weak referencing model id ", id(conshdlr.model))
    assert conshdlr.model == scip
    print(id(conshdlr.model), id(scip))

    # include pricer
    print("including pricer")
    pricer = ColoringPricer();
    scip.includePricer(pricer, "coloring", "pricer for coloring", priority=5000000)
    print("done including pricer: id", id(pricer))
    leaving()

    print("---------------start optimizing!!!!")
    scip.optimize()

    #scip = Model()
    #reader = ColReader() # should it be a proper reader?
    #scip.includeReader(reader, "colreader", "reader for col files", "col")
    #scip.readProblem(sys.argv[-1])

    #scip, x = create_sudoku()

    #scip.setBoolParam("misc/allowdualreds", False)
    #scip.setBoolParam("misc/allowdualreds", False)
    #scip.setEmphasis(SCIP_PARAMEMPHASIS.CPSOLVER)
    #scip.setPresolve(SCIP_PARAMSETTING.OFF)
    #scip.optimize()

    #if scip.getStatus() != 'optimal':
    #    print('Sudoku is not feasible!')
    #else:
    #    print('\nSudoku solution:\n')
    #    for row in range(9):
    #        out = ''
    #        for col in range(9):
    #            out += str(round(scip.getVal(x[row,col]))) + ' '
    #        print(out)

if __name__ == "__main__":
    test_main()

#class ColReader(Reader):
#    def readerread(self, file):
#        print("reading file ", file)
#        # open file
#        f = open(file, 'r')
#        # if fails: return {"result": SCIP_RESULT.READERROR}
#        # get nodes of graph
#        for line in f:
#            if "p" == line[0]:
#                nedges = int(line.split()[2])
#                break
#        G = networkx.Graph()
#        G.add_nodes_from(range(1,nedges+1))
#        # get edges of graph
#        for line in f:
#            if "e" == line[0]:
#                s = line.split()
#                G.add_edge(int(s[1]), int(s[2]))
#        print(G.nodes())
#        print(G.edges())
#        # get problem name (can't get them, don't understand python's problem with strings)
#        #name = file.rsplit('/',1)[-1][:-3]
#        #print("problem name is ", name)
#        return {"result": SCIP_RESULT.SUCCESS}
