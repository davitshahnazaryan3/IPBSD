"""
defines loss curve
"""
import numpy as np
from verifications.ealCheck import EALCheck


class LossCurve:
    # todo, add function which will optimize SLS parameters until EAL condition is met
    def __init__(self, y, lam, eal_limit):
        """
        initializes loss curve definition
        :param y: array                 Expected loss ratios
        :param lam: array               Mean annual frequencies of exceeding limit states
        :param eal_limit: float         Limit EAL value
        """
        self.y = y
        self.lam = lam
        self.eal_limit = eal_limit
        self.EAL = None
        self.loss_curve()

    def loss_curve(self):
        """
        Fitting of a refined loss curve that passes through the performance limit states and calculates the expected
        annual loss (EAL) as the area below the refined loss curve
        :return: None
        """
        coef = np.zeros(len(self.y))

        r1 = np.zeros(len(self.y))
        r2 = np.zeros(len(self.y))
        r3 = np.zeros(len(self.y))
        r1[0] = 1
        r2[0] = 1
        r3[0] = 1
        r1[1] = -np.log(self.y[0])
        r2[1] = -np.log(self.y[1])
        r3[1] = -np.log(self.y[2])
        r1[2] = -np.log(self.y[0]) ** 2
        r2[2] = -np.log(self.y[1]) ** 2
        r3[2] = -np.log(self.y[2]) ** 2

        temp1 = np.log(self.lam)
        temp2 = np.array([r1, r2, r3])
        temp3 = np.linalg.inv(temp2).dot(temp1)
        temp3 = temp3.tolist()

        coef[0] = np.exp(temp3[0])
        coef[1] = temp3[1]
        coef[2] = temp3[2]

        y_fit = np.linspace(0.01, 1., 100)
        area = []
        lambda_fit = coef[0] * np.exp(-coef[1] * np.log(y_fit) - coef[2] * np.log(y_fit) ** 2)

        y_fit = np.insert(y_fit, 0, 0.0)
        lambda_fit = np.insert(lambda_fit, 0, lambda_fit[0])

        for i in range(len(lambda_fit) - 1):
            area.append((lambda_fit[i] + lambda_fit[i + 1]) / 2 * (y_fit[i + 1] - y_fit[i]))
        self.EAL = sum(area) * 100

    def verify_eal(self):
        """
        verifies if EAL is below the limit EAL
        :return: None
        """
        eal_check = EALCheck(self.EAL, self.eal_limit)
        if eal_check.verify_eal():
            print(f"[SUCCESS] EAL condition is met! Diff.: {(self.eal_limit - self.EAL)/self.eal_limit*100:.1f}%")
        else:
            print(f"[FAILURE] EAL condition is not met! Diff.: {(self.eal_limit - self.EAL)/self.eal_limit*100:.1f}%")
