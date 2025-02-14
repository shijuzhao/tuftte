from gurobipy import Model, GRB, quicksum

class GurobiSolver:
    def __init__(self):
        self.problem = Model()
        self.problem.Params.OutputFlag = 0
        self.Sum = quicksum

    def Variable(self, lb = -GRB.INFINITY, type = None):
        if type == "Int":
            return self.problem.addVar(lb=lb, vtype=GRB.INTEGER)
        elif type == "Bool":
            return self.problem.addVar(lb=lb, vtype=GRB.BINARY)
        return self.problem.addVar(lb=lb)

    def Variables(self, shape = 1, lb = -GRB.INFINITY, type=None):
        if type == "Int":
            return self.problem.addVars(shape, lb=lb, vtype=GRB.INTEGER)
        elif type == "Bool":
            return self.problem.addVars(shape, lb=lb, vtype=GRB.BINARY)
        return self.problem.addVars(shape, lb=lb)
    
    def Maximize(self, objective):
        self.problem.setObjective(objective, GRB.MAXIMIZE)

    def Minimize(self, objective):
        self.problem.setObjective(objective, GRB.MINIMIZE)

    def Assert(self, constraint):
        self.problem.addConstr(constraint)

    def Solve(self):
        self.problem.optimize()
        if self.problem.status == GRB.Status.OPTIMAL:
            print("Optimal solution was found.")
        else:
            print("Optimal solution was not found.")
            
        return self.problem.ObjVal
    
    def Value(self, var):
        return var.x