import cvxpy as cp

class CvxpySolver:
    def __init__(self):
        self.constraints = []
        self.objective = None
        self.problem = None
        self.Sum = cp.sum

    def Variable(self, lb = None, type = None):
        if type == "Int":
            var = cp.Variable(integer = True)            
        elif type == "Bool":
            var = cp.Variable(boolean = True)
        else:
            var = cp.Variable()
        if lb is not None:
            self.constraints.append(var >= lb)
        return var

    def Variables(self, shape = 1, lb = None, type = None):
        if type == 'Int':
            var = cp.Variable(shape, integer = True)
        elif type == "Bool":
            var = cp.Variable(shape, boolean = True)
        else:
            var = cp.Variable(shape)
        if lb is not None:
            self.constraints.append(cp.min(var) >= lb)
        return var
    
    def Maximize(self, objective):
        self.objective = cp.Maximize(objective)

    def Minimize(self, objective):
        self.objective = cp.Minimize(objective)

    def Assert(self, constraint):
        self.constraints.append(constraint)

    def Solve(self):
        assert self.objective
        prob = cp.Problem(self.objective, self.constraints)
        self.problem = prob
        obj = prob.solve(solver=cp.CLARABEL)
        if prob.status == 'optimal':
            print("Optimal solution was found.")
        else:
            print("Optimal solution was not found.")
            
        return obj

    def Value(self, var):
        return var.value