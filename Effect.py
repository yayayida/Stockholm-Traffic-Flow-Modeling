import numpy as np
import pandas as pd
from tabulate import tabulate
def Effect(v_car, v_pt, v_slow, dist, carparking, transitprice, EU):
    
    # Here, calculate the vehicle kilometers traveled and externalities    
    VKT = dist * v_car
    wt = 0.01 * VKT
    acc = 0.25 * VKT
    noise = 0.081 * VKT
    emission = 0.004 * VKT
    co2 = 3.15 * 0.085 * 1.5 * VKT
    total_externalities = np.sum(wt) + np.sum(acc) + np.sum(noise) + np.sum(emission) + np.sum(co2)
    parking_rev = v_car * carparking
    transit_rev = v_pt * transitprice
    total_rev = np.sum(transit_rev) + np.sum(parking_rev)
    
    car_trips = np.sum(v_car)
    pt_trips = np.sum(v_pt)
    slow_trips = np.sum(v_slow)
    total_trips = car_trips + pt_trips + slow_trips
    car_share = (car_trips / total_trips * 100) if total_trips > 0 else 0
    pt_share = (pt_trips / total_trips * 100) if total_trips > 0 else 0
    slow_share = (slow_trips / total_trips * 100) if total_trips > 0 else 0
    
    # create the table with externalities 
    table = {
        "Car Trips":            f"{int(car_trips):,}",
        "Car Mode Share (%)":        f"{car_share:.2f}%",
        "Transit Trips":        f"{int(pt_trips):,}",
        "Transit Mode Share (%)":   f"{pt_share:.2f}%",
        "Slow Trips":           f"{int(slow_trips):,}",
        "Slow Mode Share (%)":      f"{slow_share:.2f}%",
        "Total trips":           f"{int(total_trips):,}",
        "Distance Travelled (1000 km)": f"{int(np.sum(VKT)/1000):,}",
        "Noise Cost (€)":       f"{int(np.sum(noise)):,}",
        "Waiting Time Cost (€)": f"{int(np.sum(wt)):,}",
        "Accident Cost (€)":    f"{int(np.sum(acc)):,}",
        "Emission Cost (€)":    f"{int(np.sum(emission)):,}",
        "CO2 Cost (€)":         f"{int(np.sum(co2)):,}",
        "Total Externalities (€)": f"{int(total_externalities):,}",
        "Parking Revenue (€)":  f"{int(np.sum(parking_rev)):,}",
        "Transit Revenue (€)":  f"{int(np.sum(transit_rev)):,}",
        "Total Revenue (€)":    f"{int(total_rev):,}",
        "Expected Utility":      f"{int(np.sum(EU)):,}"
    }
    
    return pd.DataFrame(table.values(), index=list(table.keys()), columns=['VALUE'])
