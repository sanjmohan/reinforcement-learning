from kaggle_environments import evaluate, make, utils
import sys


out = sys.stdout
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx")
# passing timeout configuration to make doesn't construct env properly...
# validate against 5s timeout like in kaggle leaderboard
env.configuration.timeout = env.configuration.actTimeout = 5
print("Validating - this may take a minute...")
s = env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed..."+"\n"+s)

