import numpy as np
import scipy.interpolate
from scipy.stats import multivariate_normal
from itertools import combinations
from scipy.interpolate import BSpline, UnivariateSpline
import matplotlib.pyplot as plt


def findM1(meanUt, indM, invSigM, idx, rewards, average_rewards, stdM, losst, al, idx_rows):
    nM, m = indM.shape
    Q = np.zeros(nM)
    Q1 = np.zeros(nM)

    for i in range(nM):
        Q1[i] = np.dot(np.dot(meanUt[indM[i, :] - 1], invSigM[i]), meanUt[indM[i, :] - 1].T)
        tmp = np.zeros((m, m))
        tmpPhi = invSigM[i]

        for j in range(m):
            for k in range(m):
                tmp[j][k] = np.sqrt(losst) * (np.abs(tmpPhi[j, k] * meanUt[indM[i, j] - 1] * stdM[indM[i, k] - 1]) +
                                              np.abs(tmpPhi[j, k] * meanUt[indM[i, k] - 1] * stdM[indM[i, j] - 1]) +
                                              losst * np.abs(
                            tmpPhi[j, k] * stdM[indM[i, k] - 1] * stdM[indM[i, j] - 1])) + average_rewards[idx] * (
                                    1 - al) + rewards[indM[i, k] - 1][i] * al
        Q[i] = Q1[i] + np.sum(tmp)

    return Q


#

class Client:

    def __init__(self, data, m0, m, covMat, loss, lambda_val, limit, indM, invSigM, tau):
        self.data = data.T
        self.m0 = m0
        self.m = m
        self.covMat = covMat
        self.loss = loss
        self.lambda_val = lambda_val
        self.limit = limit
        self.indM = indM
        self.invSigM = invSigM
        self.tau = tau

        self.kmax, self.dim = self.data.shape
        self.meanU0 = np.zeros(self.dim)
        self.invcovMat = np.linalg.inv(self.covMat)

        self.Q = np.zeros((1, indM.shape[0]))
        self.count = np.zeros((self.kmax - self.m0, self.dim)).T
        self.count[0] = 1
        self.runlen = self.kmax - self.m0 - self.tau
        self.rewards = np.zeros((n, nM))
        self.average_rewards = np.zeros((n,))
        self.index = np.zeros((self.m, self.kmax - self.m0), dtype=int)
        self.meanU = np.zeros((self.kmax - self.m0, self.dim))
        self.sumU = np.zeros((self.kmax - self.m0, self.dim))
        self.Omega = [None] * (self.kmax - m0)
        for j in range(self.kmax - m0):
            self.Omega[j] = np.array([])
        self.Test = np.zeros(self.kmax - self.m0)

    def play(self, j, i, al):

        if i == 0:
            self.Omega[i] = (self.invcovMat + np.linalg.inv(self.covMat))
            self.sumU[i, :] = self.meanU0 + self.data[m0 + 1, :]
            self.meanU[i, :] = self.sumU[i, :] @ np.linalg.inv(self.Omega[i])

        if i > 0:
            self.Q = findM1(self.meanU[i - 1, :], self.indM, self.invSigM, j, self.rewards, self.average_rewards,
                            np.sqrt(np.diag(np.linalg.inv(self.Omega[i - 1] + 1e-4 * np.eye(self.dim)))), self.loss(i),
                            al,
                            j)
            self.index[:, i] = self.indM[self.Q.argmax()].T

            obsM = self.data[i + self.m0 - 1, self.index[:, i] - 1]
            dataM = np.zeros(self.dim)
            dataM[self.index[:, i] - 1] = obsM

            Z = np.zeros((self.m, self.dim))
            Z[np.arange(self.m), self.index[:, i] - 1] = 1

            smallCovMat = Z @ self.covMat @ Z.T
            self.Omega[i] = self.Omega[i - 1] * (1 - self.lambda_val) + Z.T @ np.linalg.inv(smallCovMat) @ Z

            self.sumU[i] = self.sumU[i - 1] * (1 - self.lambda_val) + dataM[self.index[:, i] - 1] @ np.linalg.inv(
                smallCovMat) @ Z
            self.meanU[i] = self.sumU[i] @ np.linalg.inv(self.Omega[i] + 1e-4 * np.eye(self.dim))

        self.Test[i] = self.meanU[i] @ self.Omega[i] @ self.meanU[i].T

        nM, _ = self.indM.shape

        if i > self.tau and self.Test[i] > limit - 1:
            self.runlen = i - self.tau + 1
            return None
        Q1 = np.zeros(nM)
        for k in range(nM):
            Q1[k] = self.meanU[i, self.indM[k, :] - 1] @ self.invSigM[k] @ self.meanU[i, self.indM[k, :] - 1].T
        self.rewards[j, :] = Q1
        self.average_rewards = np.average(self.rewards, axis=1)
        return Q1


class Server:
    def __init__(self, client_count, data, m0, m, covMat, loss, lambda_val, limit, indM, invSigM, tau, al):
        self.client_count = client_count
        self.data = data
        self.m0 = m0
        self.m = m
        self.covMat = covMat
        self.loss = loss
        self.lambda_val = lambda_val
        self.limit = limit
        self.indM = indM
        self.invSigM = invSigM
        self.tau = tau
        self.al = al
        self.clients = [Client(data, m0, m, covMat, loss, lambda_val, limit, indM, invSigM, tau) for _ in
                        range(client_count)]

    def run(self):
        kmax, dim = self.data.T.shape
        nM, _ = self.indM.shape
        rewards = np.zeros((self.client_count, nM))
        for j, c in enumerate(self.clients):
            for i in range(kmax - self.m0):
                rewards[j, :] = c.play(j, i, self.al)
                if c.runlen < kmax:
                    break

        average_rewards = np.average(rewards, axis=0)
        assert average_rewards.shape == (nM,)
        return [client.index for client in self.clients], [client.runlen for client in self.clients]


def generate_data(n, T, bd, k0, bd0, nq0, k1, ndefect, delta, sigma, Tin):
    # Generate B00
    B0 = fourierbasis(nq0, n)
    B00 = B0
    sigma0 = 0.1
    Sigma0 = np.eye(B00.shape[1]) * sigma0
    mu0 = np.zeros((1, B00.shape[1]))
    theta0 = np.zeros((T, B00.shape[1]))
    # sample IC data
    Y = np.zeros((n, T))
    for i in range(T):
        theta0[i, :] = multivariate_normal.rvs(mu0[0], Sigma0, 1)

        Y[:, i] = B00.dot(theta0[i, :]) + sigma * np.random.randn(n)

    # create knots
    knots = np.concatenate((np.ones(bd), np.linspace(1, n, int(round(n / k1))), np.full(bd, n)))

    # calculate the number of knots
    nKnots = len(knots) - (bd - 1)

    # create B-spline
    kspline = BSpline(knots, np.eye(nKnots), k=bd - 1)

    # calculate B-spline basis function 
    x = np.arange(1, n + 1)
    B = kspline(x)
    B = B[:, 2:-2]
    kcoef = B.shape[1]
    Z = np.zeros((kcoef, 1))

    for i in range(ndefect):
        Z[np.random.choice(kcoef)] = 1

    D = B.dot(Z)

    for i in range(Tin + 1, T):
        Y[:, i] = Y[:, i] + delta * D.flatten()

    Y1 = Y
    return Y1


def loss(t):
    return np.sqrt(np.maximum(0, np.log((1 - (1 - lambda_) ** (t + 1)) / lambda_)))


def run_rep(i, delta, al):
    Y = generate_data(n, T, bd, k0, bd0, nq0, k1, ndefect, delta, sigma, Tin)

    result, runlen_result = Server(n, Y, m0, m, covMat, loss, lambda_, limit, indM, invSigM, tau, al).run()
    matrix[i] = np.array(runlen_result)
    print("Process " + str(i) + " completed")
    print("runlen of each client:", matrix[i])
    f = open("delta_and_alpha.txt", "a")

    f.write("delta:" + str(delta) + " alpha:" + str(al) + "\n")
    f.write("Process " + str(i) + " completed\n")
    f.write("runlen of each client:" + str(matrix[i]) + "\n")
    f.close()
    return result


def run_simulation_no_pool(nrep):
    for dt in delta_list:
        delta = dt
        for al in alpha:

            print("delta:{} alpha:{}".format(delta, al))
            f = open("delta_and_alpha.txt", "a")
            res = [run_rep(i, delta, al) for i in range(nrep)]
            mean_ = np.mean(matrix, axis=0)
            std_ = np.std(matrix, axis=0)
            for i in range(n):
                print("client:{}".format(i) + " ARL_oc:{}".format(mean_[i]) + " sdrl_oc:{}".format(std_[i]))
                f.write("client:{}".format(i) + " ARL_oc:{}".format(mean_[i]) + " sdrl_oc:{}".format(std_[i]) + "\n")
            f.close()
    return TTest, RRunlength, res


def fourierbasis(nq, ntime):
    point = np.linspace(-np.pi, np.pi, ntime)
    norder = np.ceil(nq / 2).astype(int)
    basis = np.zeros((ntime, nq))

    for i in range(1, norder + 1):
        basis[:, 2 * i - 2] = np.cos(i * point)
        if 2 * i <= nq:
            basis[:, 2 * i - 1] = np.sin(i * point)

    return basis


n = 10  # Image size n*n
T = 2000  # The Total time length
bd = 5  # How smooth the defect is (Order of Spline)
k0 = 20  # scale of the background
bd0 = 3
nq0 = 2
k1 = 2  # scale of the defect
ndefect = 1  # How many defect in the system
delta_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
sigma = 0.05  # The noise level
Tin = 1  # When the change occur
Y = np.zeros((n, n, T))  # Generated Videos
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # The alpha parameter of the loss function
limit = 11.25
nrep = 10
tau = 0
m = 4  # budget of pixels that can be observed for each image
lambda_ = 0.1
m0 = 0
TTest = np.zeros((nrep, T - m0))
RRunlength = np.zeros((nrep, n))
count1 = np.zeros((nrep, n))
meanU = count1
B0 = fourierbasis(nq0, n)
B00 = B0.copy()
sigma0 = 0.1
Sigma0 = np.eye(B00.shape[1]) * sigma0
eye_n = np.eye(n)
covMat = np.eye(n) * sigma * sigma + sigma0 * np.dot(np.dot(B00, np.eye(B00.shape[1])), B00.T)
combinations_list = list(combinations(range(1, n + 1), m))
indM = np.array(combinations_list)
nM = len(indM)
invSigM = []
matrix = np.zeros((nrep, n))

for i in range(nM):
    
    subset_indices = indM[i, :] - 1

    covMat_subset = covMat[subset_indices][:, subset_indices]

    inv_covMat_subset = np.linalg.inv(covMat_subset)

    invSigM.append(inv_covMat_subset)

if __name__ == "__main__":
    TTest, RRunlength, res = run_simulation_no_pool(nrep)
    # res = run_simulation_no_pool(nrep)
    # print(res)
