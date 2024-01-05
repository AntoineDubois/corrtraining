import matplotlib.pyplot as plt
import numpy as np

loss = np.loadtxt("./build/loss.txt")
avsc = np.loadtxt("./build/avscore.txt")

f, ax = plt.subplots(2)
ax[0].plot(loss, label="loss")
ax[0].set_title("loss")
ax[1].plot(avsc, label="av score")
ax[1].set_title("av score")
ax[1].axhline(y = 0.0, color = 'r', linestyle = '-') 
plt.show()

#S = np.random.uniform(size = (10, 10))
#S = S.T @ S
#print(np.all(S == S.T))
#print(np.all(np.linalg.eig(S)[0] >= 0))
#print(S)