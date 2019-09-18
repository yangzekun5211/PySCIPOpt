from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE
import sys
import matplotlib.pyplot as plt

pbs  = []
tpbs = []
dbs  = []
tdbs = []

class NodeSolvedEvent(Eventhdlr):

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        pbs.append(self.model.getPrimalbound())
        tpbs.append(self.model.getSolvingTime())
        dbs.append(dbs[-1])
        tdbs.append(self.model.getSolvingTime())

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND:
            pbs.append(self.model.getPrimalbound())
            tpbs.append(self.model.getSolvingTime())
        else:
            assert event.getType() == SCIP_EVENTTYPE.LPSOLVED
            dbs.append(self.model.getDualbound())
            tdbs.append(self.model.getSolvingTime())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python primal_dual.py <instance> [optimum]")
        exit(0)

    # create model
    m = Model("primal_dual")

    # create and add an event handler
    eventhdlr = NodeSolvedEvent()
    m.includeEventhdlr(eventhdlr, "primal_dual_events", "python event handler to catch LPSOLVED and BESTSOLFOUND events")

    # set settings
    m.setIntParam("separating/maxroundsroot", 5)
    m.setIntParam("separating/maxrounds", 5)
    m.setRealParam("limits/gap", 1e-3)

    # optimize given instance
    m.readProblem(sys.argv[1])
    m.optimize()
    solvingtime = m.getSolvingTime()

    del m

    plt.step(tpbs, pbs, linewidth=4.0)
    plt.plot(tdbs, dbs, linewidth=4.0)

    if len(sys.argv) == 3:
        optimum = float(sys.argv[2])
        plt.plot([0,solvingtime],[optimum,optimum], '--', linewidth=4.0, color='black')
        plt.legend(["primal bound", "dual bound", "optimum"], fontsize='x-large')
    else:
        plt.legend(["primal bound", "dual bound"], fontsize='x-large')

    plt.grid(True)
    plt.xlabel("solving time", fontsize='x-large')
    plt.ylabel("primal/dual bound", fontsize='x-large')
    plt.show()
