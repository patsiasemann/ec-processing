import numpy as np

# Import the roughness data with semicolon delimiter
file_path = "Data/SILVEX2_Silvia2_roughness_ffp_data.CSV"
df_z0 = pd.read_csv(file_path, sep=';', low_memory=False)

kappa = 0.4  # von Kármán constant
z1 = 1.1  # measurement height 1 (m)
z2 = 2.1  # measurement height 2 (m)
z3 = 3.1  # measurement height 3 (m)

# Compute aerodynamic roughness length z_0 for each level
# Formula (assuming neutral conditions): u_z = (u* / kappa) * ln(z/z_0)
# Solving for z_0: z_0 = z * exp(-u_z * kappa / u*)

df_z0['z0_1m'] = z1 * np.exp(-df_z0['wind_speed_1m'] * kappa / df_z0['u*_1m'])
df_z0['z0_2m'] = z2 * np.exp(-df_z0['wind_speed_2m'] * kappa / df_z0['u*_2m'])
df_z0['z0_3m'] = z3 * np.exp(-df_z0['wind_speed_3m'] * kappa / df_z0['u*_3m'])

df_z0['z0_stable_1m'] = z1 * np.exp(-df_z0['wind_speed_1m'] * kappa / df_z0['u*_1m'] + 4.7 * df_z0['(z-d)/L_1m'])
df_z0['z0_stable_2m'] = z2 * np.exp(-df_z0['wind_speed_2m'] * kappa / df_z0['u*_2m'] + 4.7 * df_z0['(z-d)/L_2m'])
df_z0['z0_stable_3m'] = z3 * np.exp(-df_z0['wind_speed_3m'] * kappa / df_z0['u*_3m'] + 4.7 * df_z0['(z-d)/L_3m'])

# Display basic statistics for the computed roughness lengths
print("Aerodynamic Roughness Length Statistics:\n")
print(df_z0[['z0_1m', 'z0_2m', 'z0_3m']].describe(), "\n")
print(df_z0[['z0_stable_1m', 'z0_stable_2m', 'z0_stable_3m']].describe())
print("\n\nFirst few values:")
print(df_z0[['wind_speed_1m', 'u*_1m', 'z0_1m',
                'wind_speed_2m', 'u*_2m', 'z0_2m',
                'wind_speed_3m', 'u*_3m', 'z0_3m']].head(10))