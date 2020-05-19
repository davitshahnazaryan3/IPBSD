"""
checks for errors in the input file
"""


class ErrorCheck:
    def __init__(self, df):

        # PLS feature
        pls_list = ['OLS', 'SLS', 'ULS', 'CLS']
        for i in df['PLS']:
            if df['PLS'][i] not in pls_list:
                raise ValueError('Performance limit state not supplied properly')

        # ELR, TR, aleatory features
        if len(df['ELR']) != len(df['PLS']):
            raise ValueError('ELR not supplied properly')
        if len(df['TR']) != len(df['PLS']):
            raise ValueError('TR not supplied properly')
        if len(df['aleatory']) != len(df['PLS']):
            raise ValueError('LS uncertainties not supplied properly')

        # ELR feature
        for i, j in df['ELR'].items():
            if j > 1:
                raise ValueError('ELR cannot be higher than 1')
            if i > 0:
                if df['ELR'][i] < df['ELR'][i - 1]:
                    raise ValueError('ELR of higher LS cannot be smaller than at lower LS')

        # TR feature
        for i, j in df['TR'].items():
            if i > 0:
                if df['TR'][i] < df['TR'][i - 1]:
                    raise ValueError('TR of higher LS cannot be smaller than at lower LS')

        # bldg_ch feature
        if len(df['bldg_ch']) != 3:
            raise ValueError('Building characteristics are not supplied properly')

        # mode_red feature
        if len(df['mode_red']) != 1:
            raise ValueError('Mode reduction factor should have one entry value')

        # fy feature
        if len(df['fy']) != 1:
            raise ValueError('Yield strain of reinforcement is not supplied properly')

        # n_seismic_frames feature
        if len(df['n_seismic_frames']) != 1:
            raise ValueError('Number of seismic frames is not supplied properly')

        # n_gravity_frames feature
        if len(df['n_gravity_frames']) != 1:
            raise ValueError('Number of gravity frames is not supplied properly')
