"""
    low_voltage_ride_through(t_range)

Calculates a time series for a "low voltage fault-ride through limiting curve" based on the one given in:
"Probabilistic Stability Assessment for Active Distribution Grids" (See figure 5)
https://arxiv.org/abs/2106.09624
"""
const t_range_low = [0.0, 0.15, 3.0, 60.0 - 10^-5, 60.0, 1000.0] 
const limits_low = [0.15, 0.15, 0.85, 0.85, 0.9, 0.9]

const low_voltage_ride_through = linear_interpolation(t_range_low, limits_low)

"""
    high_voltage_ride_through(t_range)

Calculates a time series for a "high voltage fault-ride through limiting curve" based on the one given in:
"E VDE-AR-N 4130:2017-09, Technical requirements for the connection and operation of customer installations to the extra high voltage network (TAR extra high voltage)" (See figure 11)
"""
const t_range_high = [0.0, 0.1 - 10^-5, 0.1, 60.0 - 10^-5, 60.0, 1000.0] 
const limits_high = [1.3, 1.3, 1.2, 1.2, 1.1, 1.1]

const high_voltage_ride_through = linear_interpolation(t_range_high, limits_high)