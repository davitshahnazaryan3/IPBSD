"""
second-order hazard fitting to an input pickle
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimization


class HazardFit:
    def __init__(self, directory, filename, haz_fit=1, pflag=False, save_data=True):
        """
        init hazard fitting tool
        :param directory: str                                   Hazard file directory
        :param filename: str                                    Hazard file name
        :param haz_fit: int                                     Hazard fitting function to use (1, 2, 3)
        :param pflag: bool                                      Plot info or not
        :param save_data: bool                                  Save fitted data or not
        :return: None
        """
        self.directory = directory
        self.filename = filename
        self.pflag = pflag
        self.haz_fit = haz_fit
        self.ITERATOR = np.array([0, 3, 5])
        self.saveData = save_data
        self.hazard_fit, self.s_fit = self.run_fitting(self.haz_fit, self.read_hazard())

    def read_hazard(self):
        """
        reads provided hazard data and plots them
        :return: dict                                           True hazard data
        """
        with open(self.directory/self.filename, 'rb') as file:
            [im, s, apoe] = pickle.load(file)
            im = np.array(im)
            s = np.array(s)
            apoe = np.array(apoe)
        data = {'im': im, 's': s, 'apoe': apoe}

        if self.pflag:
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            self.plotting(version='Original', data=data)

        return data

    @staticmethod
    def plotting(version, data, tag=None, s_fit=None, H_fit=None):
        """
        Plots the input true hazard
        :param version: str                                     Original or Fitted version to plot
        :param data: dict                                       Hazard data
        :param tag: int                                         Record tag
        :param s_fit: array                                     Fitted hazard intensity measures
        :param H_fit: array                                     Fitted hazard probabilities
        :return: None
        """
        if version == "Original":
            for i in range(len(data['im'])):
                plt.loglog(data['s'][i], data['apoe'][i])
            plt.grid(True, which="both", ls="--")
            plt.xlim([1e-3, 100])
            plt.ylim([1e-4, 1])
        elif version == "Fitted":
            plt.loglog(data['s'][tag], data['apoe'][tag])
#            self.data['s'][self.tag],self.data['apoe'][self.tag],'-',
            plt.loglog(s_fit, H_fit, '--')
            plt.grid(True, which="both", ls="--")
            plt.xlim([1e-3, 100])
            plt.ylim([1e-6, 1])
        else:
            raise ValueError('[EXCEPTION] Wrong version of plotting!')

    def run_fitting(self, haz_fit, data):
        """
        Runs the fitting function
        :param haz_fit: bool                                    Hazard fitting function to run
        :param data: dict                                       True hazard data
        :return: dataframe, array                               Fitted Sa and H of the hazard
        """
        if haz_fit == 1:
            hazard_fit, s_fit = self.my_fitting(data)
        elif haz_fit == 2:
            hazard_fit, s_fit = self.scipy_fitting(data)
        elif haz_fit == 3:
            hazard_fit, s_fit = self.leastsq_fitting(data)
        else:
            raise ValueError('[EXCEPTION] Wrong fitting function!')
        return hazard_fit, s_fit

    def record_data(self, info):
        """
        Saves the fitting data into pickles
        :param info: dict               Fitted hazard data
        :return: None
        """
        hazard_data = {'hazard_fit': info['hazard_fit'], 's': info['s_fit'], 'T': info['T']}
        with open(self.directory/f"coef_{self.filename}", 'wb') as handle:
            pickle.dump(info['coefs'], handle)
        with open(self.directory/f"fit_{self.filename}", 'wb') as handle:
            pickle.dump(hazard_data, handle)

    def generate_fitted_data(self, im, coefs, hazard_fit, s_fit):
        """
        Generates dictionary for saving hazard data
        :param im: numpy array          Intensity measures
        :param coefs: DataFrame         2nd-order fit coefficients
        :param hazard_fit: dataframe    H of the fitted hazard
        :param s_fit: array             Sa of the fitted hazard
        :return: dict                   Fitted hazard data
        """
        T = np.zeros(len(im))
        for t in range(len(im)):
            try: T[t] = im[t].replace('SA(', '').replace(')', '')
            except: T[t] = 0.0

        info = {'hazard_fit': hazard_fit, 's_fit': s_fit, 'T': T, 'coefs': coefs}

        return info

    def my_fitting(self, data):
        """
        Hazard fitting function my version
        :param data: dict               True hazard data
        :return: dataframe, array       Fitted H and Sa of the hazard
        """
        print("[FITTING] Hazard MyVersion")
        im = data['im']
        s = data['s']
        apoe = data['apoe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))

        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=200)
        # Fitting the hazard curves
        for tag in range(len(im)):
            coef = np.zeros(3)
            # select iterator depending on where we want to have a better fit
            iterator = self.ITERATOR
            r = np.zeros((len(iterator), len(iterator)))
            a = np.zeros(len(iterator))
            cnt = 0
            for i in iterator:
                r_temp = np.array([1])
                for j in range(1, len(iterator)):
                    r_temp = np.append(r_temp, -np.power(np.log(s[tag][i]), j))
                r[cnt] = r_temp
                a[cnt] = apoe[tag][i]
                del r_temp
                cnt += 1
            temp1 = np.log(a)
            temp2 = np.linalg.inv(r).dot(temp1)
            temp2 = temp2.tolist()
            coef[0] = np.exp(temp2[0])
            coef[1] = temp2[1]
            coef[2] = temp2[2]
            H_fit = coef[0]*np.exp(-coef[2]*np.power(np.log(s_fit), 2) -
                                   coef[1]*np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = coef
            tag = tag
            if self.pflag:
                if im[tag] == 'PGA' or im[tag] == 'SA(0.7)':
                    self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.saveData:
            self.record_data(info)

        return hazard_fit, s_fit

    def scipy_fitting(self, data):
        """
        Hazard fitting function by scipy library
        :param data: dict               True hazard data
        :return: dataframe, array       Fitted H and Sa of the hazard
        """
        print("[FITTING] Hazard Scipy")
        im = data['im']
        s = data['s']
        apoe = data['apoe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0, 0, 0])
        sigma = np.array([1.0]*len(s[0]))

        def func(x, a, b, c):
            return a*np.exp(-c*np.power(np.log(x), 2)-b*np.log(x))
        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        for tag in range(len(im)):
            p, pcov = optimization.curve_fit(func, s[tag], apoe[tag], x0, sigma)
            H_fit = p[0]*np.exp(-p[2]*np.power(np.log(s_fit),2) -
                                p[1]*np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = p
            tag = tag
            if self.pflag:
                self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.saveData:
            self.record_data(info)

        return hazard_fit, s_fit

    def leastsq_fitting(self, data):
        """
        Hazard fitting function least squares method
        :param data: dict               True hazard data
        :return: dataframe, array       Fitted H and Sa of the hazard
        """
        print("[FITTING] Hazard leastSquare")
        im = data['im']
        s = data['s']
        apoe = data['apoe']
        s_fit = np.linspace(min(s[0]), max(s[0]), 1000)
        hazard_fit = pd.DataFrame(np.nan, index=range(len(s_fit)), columns=list(im))
        coefs = pd.DataFrame(np.nan, index=['k0', 'k1', 'k2'], columns=list(im))
        x0 = np.array([0, 0, 0])

        def func(x, s, a):
            return a-x[0]*np.exp(-x[2]*np.power(np.log(s), 2)-x[1]*np.log(s))

        if self.pflag:
            fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
        for tag in range(len(im)):
            p = optimization.leastsq(func, x0, args=(s[tag], apoe[tag]))[0]
            H_fit = p[0]*np.exp(-p[2]*np.power(np.log(s_fit), 2) -
                                p[1]*np.log(s_fit))
            hazard_fit[im[tag]] = H_fit
            coefs[im[tag]] = p
            tag = tag
            if self.pflag:
                self.plotting(version='Fitted', data=data, tag=tag, s_fit=s_fit, H_fit=H_fit)

        info = self.generate_fitted_data(im, coefs, hazard_fit, s_fit)

        if self.saveData:
            self.record_data(info)

        return hazard_fit, s_fit
