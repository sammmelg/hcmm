import pandas as pd
import numpy as np
import WesternMeteorPyLib.wmpl.Utils.AtmosphereDensity as ad
import WesternMeteorPyLib.wmpl.MetSim.MetSimErosion as metsim
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def calc_kb(beg_atmos_dens, initial_vel, zenith_dist):
    """
    Compute Kb value for each row in the dataframe from using equation in Ceplecha 1988.

    Parameters
    ----------
    beg_atmos_dens : pandas.Series
        Meteor's beginning atmospheric density (gm/cm^3).
    initial_vel : pandas.Series
        Meteor's initial velocity (cm/sec).
    zenith_dist : pandas.Series
        Meteor's zenith angle (degrees).

    Returns
    -------
    kb_values : pandas.Series
        Kb value for each row in the dataframe, offset by -0.10 (Cordonnier et al. (2024)).
    """
    kb_values = (np.log10(beg_atmos_dens) + (2.5 * np.log10(initial_vel)) -
                 (0.5 * np.log10(np.cos(np.radians(zenith_dist)))))

    return kb_values - 0.10


def compute_row(idx, row, zR_val):
    """
    Compute beginning atmospheric density and beginning energy
    received for each row in the dataframe using wmpl.

    Parameters
    ----------
    idx : int
        Row index in the dataframe. Used to put results back in the same order as the original dataframe.
    row : pandas.Series
        A single row of values from the dataframe.
    zR_val : pandas.Series
        Meteor's zenith angle (degrees).

    Returns
    -------
    idx : pandas.Series
        Kb value for each row in the dataframe, offset by -0.10 (Cordonnier et al. (2024)).
    beg_atmos_dens : pandas.Series
        Meteor's beginning atmospheric density (kg/m^3).
    beg_energy : pandas.Series
        Meteor's energy received before ablation (MJ/m^2).
    """
    try:
        # Beginning atmospheric density (Height: meters)
        beg_atmos_dens = ad.getAtmDensity(np.radians(row['LatBeg']), np.radians(row['LonBeg']),
                                               row['HtBeg'] * 1000, row['Beginning_JD'])

        # Beginning Energy received (Heights: meters, Velocity: m/s)
        beg_ml_const = metsim.Constants()
        beg_ml_const.erosion_height_start = row['HtBeg'] * 1000
        beg_ml_const.v_init = row['Vinit'] * 1000
        beg_ml_const.zenith_angle = np.radians(zR_val)
        beg_ml_const.m_init = row['Mass_kg']
        poly = ad.fitAtmPoly(np.radians(row['LatBeg']), np.radians(row['LonBeg']),
                             beg_ml_const.erosion_height_start, beg_ml_const.h_init, row['Beginning_JD'])
        beg_ml_const.dens_co = poly
        beg_energy = metsim.energyReceivedBeforeErosion(beg_ml_const)[0] / 1e6

        return idx, beg_atmos_dens, beg_energy

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        return idx, np.nan, np.nan


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, help="File path to raw data")
    parser.add_argument("-savefile", type=str, help="Filename for .csv of processed data")
    args = parser.parse_args()

    # Load data
    meteor_df = pd.read_csv(args.path, low_memory=False)

    # Drop unnecessary parameters
    drop_params = ['IAU_number', 'Sollon', 'AppLST', 'Rageo', 'RAgeo_err', 'DECgeo', 'DECgeo_err', 'LAMgeo',
                   'LAMgeo_err', 'BETgeo', 'BETgeo_err', 'Vgeo', 'Vgeo_err', 'LAMhel', 'LAMhel_err', 'BEThel',
                   'BEThel_err', 'Vhel', 'Vhel_err', 'a', 'a_err', 'e', 'e_err', 'i', 'i_err', 'peri', 'peri_err',
                   'node', 'node_err', 'Pi', 'Pi_err', 'b', 'b_err', 'q', 'q_err', 'f', 'f_err', 'M', 'M_err',
                   'Q', 'Q_err', 'n', 'n_err', 'T', 'T_err', 'Raapp', 'RAapp_err', 'DECapp', 'DECapp_err',
                   'Qc', 'MedianFitErr', 'Num', 'Participating', 'TisserandJ_err', 'Azim_E_err', 'Elev_err',
                   'Vinit_err', 'Vavg_err', 'LatBeg_err', 'LonBeg_err', 'HtBeg_err', 'LatEnd_err',
                   'LonEnd_err', 'HtEnd_err', 'Beg_in', 'End_in']
    meteor_df = meteor_df.drop(drop_params, axis=1)

    # Zenith angles
    zR = 90 - meteor_df['Elev']

    # Prepare args for each row
    rows = list(meteor_df.iterrows())
    args_list = [(i, row, zR.loc[i]) for i, row in rows]

    # Compute values that are necessary via parallel processing
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_row, i, row, zr) for i, row, zr in args_list]

        # Progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows",
                           mininterval=10.0):
            results.append(future.result())

    # Sort and unpack results
    results.sort()
    _, begin_atmDens, begin_energy_rec = zip(*results)

    # Post-process
    meteor_df.insert(10, 'Deceleration', (meteor_df['Vinit'] - meteor_df['Vavg']) / meteor_df['Duration'])
    meteor_df.insert(16, 'AtmDensBeg', begin_atmDens)
    meteor_df.insert(17, 'TrailLength', meteor_df['Duration'] * meteor_df['Vavg'])
    meteor_df.insert(18, 'EnergyRecBeg', begin_energy_rec)
    meteor_df.insert(19, 'ZenithAngle', zR)

    # Drop remaining unnecessary columns
    meteor_df = meteor_df.drop(['TisserandJ', 'Beginning_JD', 'Azim_E', 'Beginning', 'LatBeg',
                                'LonBeg', 'LatEnd', 'LonEnd', 'Peak_Ht'], axis=1)

    # Kb
    init_vel = meteor_df['Vinit'] * 1e5
    kb = calc_kb(meteor_df['AtmDensBeg'] / 1e3, init_vel, zR)
    meteor_df.insert(meteor_df.shape[1], 'Kb', kb)

    # Add IAU code to the dataframe
    iau_code = meteor_df.pop('IAU_code')
    meteor_df['IAU_code'] = iau_code

    # Save
    meteor_df.to_csv(args.savefile, index=False)