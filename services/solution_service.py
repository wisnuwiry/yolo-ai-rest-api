import csv

class SolutionService:
    def __init__(self, solutions_file="data/solutions.csv"):
        self.solutions_file = solutions_file
        self.solutions = self._load_solutions()

    def _load_solutions(self):
        solutions = {}
        with open(self.solutions_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = (row["plant_type"], row["disease"])
                solutions[key] = {
                     "plant_type": row["plant_type"],
                     "disease": row["disease"],
                     "disease_label": row["disease_label"],
                     "solution": row["solution"],
                }
                
        return solutions

    def get_solution(self, plant_type, disease):
        solution_data = self.solutions.get((plant_type, disease))
        if solution_data:
            return solution_data["solution"]
        return "Solution not found."
    
    def get_solution_data(self, plant_type, disease):
        return self.solutions.get((plant_type, disease), None)
