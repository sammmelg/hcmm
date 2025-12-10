import pandas as pd
import numpy as np
import WesternMeteorPyLib.wmpl.Utils.AtmosphereDensity as ad
import WesternMeteorPyLib.wmpl.MetSim.MetSimErosion as metsim
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def calc_kb(beg_atmos_dens, initial_vel, zenith_dist):
    kb_values = (np.log10(beg_atmos_dens) + (2.5 * np.log10(initial_vel)) -
                 (0.5 * np.log10(np.cos(np.radians(zenith_dist)))))

    return kb_values - 0.10


def compute_row(idx, row, zR_val):
    try:
        # Beginning atmospheric density
        begin = ad.getAtmDensity(np.radians(row['LatBeg']), np.radians(row['LonBeg']), row['HtBeg'] * 1000,
                                 row['Beginning_JD'])

        # Peak atmospheric density
        lat_peak = (np.radians(row['LatBeg']) + np.radians(row['LatEnd'])) / 2
        lon_peak = (np.radians(row['LonBeg']) + np.radians(row['LonEnd'])) / 2
        peak = ad.getAtmDensity(lat_peak, lon_peak, row['Peak_Ht'] * 1000,
                                row['Beginning_JD'])

        # End atmospheric density
        end = ad.getAtmDensity(np.radians(row['LatEnd']), np.radians(row['LonEnd']), row['HtEnd'] * 1000,
                               row['Beginning_JD'])

        # Beginning Energy received
        beg_ml_const = metsim.Constants()
        beg_ml_const.erosion_height_start = row['HtBeg'] * 1000
        beg_ml_const.v_init = row['Vinit'] * 1000
        beg_ml_const.zenith_angle = np.radians(zR_val)
        beg_ml_const.m_init = row['Mass_kg']

        poly = ad.fitAtmPoly(np.radians(row['LatBeg']), np.radians(row['LonBeg']),
                             beg_ml_const.erosion_height_start, beg_ml_const.h_init, row['Beginning_JD'])
        beg_ml_const.dens_co = poly

        beg_energy = metsim.energyReceivedBeforeErosion(beg_ml_const)[0] / 1e6

        # Peak Energy received
        peak_ml_const = metsim.Constants()
        peak_ml_const.erosion_height_start = row['Peak_Ht'] * 1000
        peak_ml_const.v_init = ((row['Vinit'] + row['Vavg']) / 2) * 1000
        peak_ml_const.zenith_angle = np.radians(zR_val)
        peak_ml_const.m_init = row['Mass_kg']

        poly = ad.fitAtmPoly(lat_peak, lon_peak,
                             peak_ml_const.erosion_height_start, peak_ml_const.h_init, row['Beginning_JD'])
        peak_ml_const.dens_co = poly

        peak_energy = metsim.energyReceivedBeforeErosion(peak_ml_const)[0] / 1e6

        # End Energy received
        end_ml_const = metsim.Constants()
        end_ml_const.erosion_height_start = row['HtEnd'] * 1000
        end_ml_const.v_init = row['Vavg'] * 1000
        end_ml_const.zenith_angle = np.radians(zR_val)
        end_ml_const.m_init = row['Mass_kg']

        poly = ad.fitAtmPoly(np.radians(row['LatEnd']), np.radians(row['LonEnd']),
                             end_ml_const.erosion_height_start, end_ml_const.h_init, row['Beginning_JD'])
        end_ml_const.dens_co = poly

        end_energy = metsim.energyReceivedBeforeErosion(end_ml_const)[0] / 1e6

        return idx, begin, peak, end, beg_energy, peak_energy, end_energy

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        return idx, np.nan, np.nan, np.nan, np.nan


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-year", type=str, help="Year as a string")
    parser.add_argument("-mininterval", type=float, default=0.1, help="Minimum interval in seconds")
    args = parser.parse_args()

    # Load data
    year = args.year
    meteor_df = pd.read_csv(f'/Users/shemmelg/grad_school/ms_class_ml/locams_testing/raw_locams_meteor_data.csv',
                            low_memory=False)
    meteor_df = meteor_df[meteor_df.Beg_in.str.contains('True')]
    meteor_df = meteor_df[meteor_df.End_in.str.contains('True')]

    print(f'The {year} dataset is {len(meteor_df)} entries long!')

    # Drop unnecessary parameters
    drop_params = ['IAU_number', 'Sollon', 'AppLST', 'Rageo', 'RAgeo_err', 'DECgeo', 'DECgeo_err', 'LAMgeo',
                   'LAMgeo_err', 'BETgeo',
                   'BETgeo_err', 'Vgeo', 'Vgeo_err', 'LAMhel', 'LAMhel_err', 'BEThel', 'BEThel_err', 'Vhel', 'Vhel_err',
                   'a', 'a_err', 'e', 'e_err', 'i', 'i_err', 'peri', 'peri_err', 'node', 'node_err', 'Pi', 'Pi_err',
                   'b',
                   'b_err', 'q', 'q_err', 'f', 'f_err', 'M', 'M_err', 'Q', 'Q_err', 'n', 'n_err', 'T', 'T_err', 'Raapp',
                   'RAapp_err', 'DECapp', 'DECapp_err', 'Qc', 'MedianFitErr', 'Num', 'Participating', 'TisserandJ_err',
                   'Azim_E_err', 'Elev_err', 'Vinit_err', 'Vavg_err', 'LatBeg_err', 'LonBeg_err', 'HtBeg_err',
                   'LatEnd_err',
                   'LonEnd_err', 'HtEnd_err', 'Beg_in', 'End_in']
    meteor_df = meteor_df.drop(drop_params, axis=1)

    # Zenith angles
    zR = 90 - meteor_df.Elev

    # Prepare args for each row
    rows = list(meteor_df.iterrows())
    args_list = [(i, row, zR.loc[i]) for i, row in rows]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_row, i, row, zr) for i, row, zr in args_list]

        # Progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows",
                           mininterval=args.mininterval):
            results.append(future.result())

    # Sort and unpack results
    results.sort()
    _, begin_atmDens, peak_atmDens, end_atmDens, begin_energy_rec, peak_energy_rec, end_energy_rec = zip(*results)

    # Post-process
    meteor_df.insert(10, 'Deceleration', meteor_df.Vinit - meteor_df.Vavg)
    meteor_df.insert(16, 'AtmDensBeg', begin_atmDens)
    meteor_df.insert(17, 'AtmDensPeak', peak_atmDens)
    meteor_df.insert(18, 'AtmDensEnd', end_atmDens)
    meteor_df.insert(20, 'TrailLength', meteor_df.Duration * meteor_df.Vavg)
    meteor_df.insert(21, 'EnergyRecBeg', begin_energy_rec)
    meteor_df.insert(22, 'EnergyRecPeak', peak_energy_rec)
    meteor_df.insert(23, 'EnergyRecEnd', end_energy_rec)
    meteor_df.insert(24, 'ZenithAngle', zR)

    # meteor_df.loc[meteor_df.Deceleration < 0, 'Deceleration'] = np.nan
    # meteor_df.loc[meteor_df.Mass_kg <= 0, 'Mass_kg'] = np.nan

    meteor_df = meteor_df.drop(['TisserandJ', 'Beginning_JD', 'Azim_E', 'Beginning', 'LatBeg',
                                'LonBeg', 'LatEnd', 'LonEnd', 'Peak_Ht'], axis=1)

    # Kb
    init_vel = meteor_df.Vinit * 1e5
    kb = calc_kb(meteor_df.AtmDensBeg / 1000, init_vel, zR)
    meteor_df.insert(meteor_df.shape[1], 'Kb', kb)

    # Add IAU code to the dataframe
    iau_code = meteor_df.pop('IAU_code')
    meteor_df['IAU_code'] = iau_code

    # Save
    meteor_df.to_csv(f'/Users/shemmelg/grad_school/ms_class_ml/locams_testing/'
                     f'{year}_gmn_labeled_meteor_data_energy_wNum_PARALLELIZED.csv', index=False)