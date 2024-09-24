from utils.config import get_render_config
from solver.mainsolver import Solver
import torch
from solver.tostagesolver import toStageSolver

# nohup f:/anaconda/envs/c-cnn/pythonw.exe h:/code/dual-modal-fusion/test.py
if __name__ == "__main__":
    torch.manual_seed(3407)
    cfg = get_render_config("config.yml")
    solver = Solver(cfg)
    # solver = toStageSolver(cfg)
    # solver.visualize_extract()
    # solver.visualize_deal()
    solver.run()
    # solver.proof()
