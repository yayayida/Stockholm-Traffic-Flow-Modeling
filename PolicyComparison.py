"""
Policy Comparison Script - Transit Waiting Time Intervention

This script demonstrates how modifying waiting times on specific transit links
can improve system-wide metrics such as Expected Utility (EU) and public transit share.

Policy: Reduce waiting times on high-frequency transit corridors (T-bana connections)
to make public transit more attractive and competitive with car travel.
"""

import numpy as np
import pandas as pd
import copy
from tabulate import tabulate

import Landuse
import Network
import Assignment
from Effect import Effect


def apply_transit_policy(G_pt, policy_config):
    """
    Apply a policy to modify waiting times on specific transit links.
    
    Parameters:
        G_pt: Transit network graph
        policy_config: Dictionary mapping (origin, destination) tuples to new wait_time values
                      or a reduction factor if value is between 0 and 1
    
    Returns:
        Modified transit network graph
    """
    G_pt_policy = copy.deepcopy(G_pt)
    
    for (origin, dest), wait_time_value in policy_config.items():
        if G_pt_policy.has_edge(origin, dest):
            # If value is between 0 and 1, treat as a reduction factor
            if 0 < wait_time_value < 1:
                original_wait = G_pt_policy.edges[origin, dest]['wait_time']
                new_wait = original_wait * wait_time_value
                G_pt_policy.edges[origin, dest]['wait_time'] = new_wait
                print(f"  {origin} → {dest}: {original_wait:.1f} min → {new_wait:.1f} min (reduced by {(1-wait_time_value)*100:.0f}%)")
            else:
                # Absolute value
                original_wait = G_pt_policy.edges[origin, dest]['wait_time']
                G_pt_policy.edges[origin, dest]['wait_time'] = wait_time_value
                print(f"  {origin} → {dest}: {original_wait:.1f} min → {wait_time_value:.1f} min")
        else:
            print(f"  Warning: Edge {origin} → {dest} not found in transit network")
    
    return G_pt_policy


def run_scenario(scenario_name, G_car, G_pt, landuse, zones, param, param_carown, 
                 vot_car, vot_in, vot_wait, cost_km, carparking, transitprice, 
                 penalty_intra_car, maxiter=50, maxdiff=0.01, smoothing=0.2):
    """
    Run a complete scenario (baseline or policy) and return results.
    
    Returns:
        Dictionary containing results including v_car, v_pt, v_slow, EU, dist, etc.
    """
    print(f"\n{'='*80}")
    print(f"Running Scenario: {scenario_name}")
    print(f"{'='*80}\n")
    
    # Origin and Destination dictionary
    origin = dict(zip(landuse.area, landuse.index))
    destination = origin.copy()
    
    # Initialize with uniform demand
    demand = np.ones((len(landuse), len(landuse)))
    
    # Initial assignment
    print("Initial network setup...")
    G_start_car = Assignment.RouteAssignment(np.asarray(demand), G_car, origin, destination, 
                                            vot_car, cost_km, penalty_intra_car)
    G_start_pt = Assignment.TransitAssignment(np.asarray(demand), G_pt, origin, destination, 
                                             vot_in, vot_wait)
    
    car_time_0, car_cost_0, dist = Assignment.Skim_car(G_start_car, origin, destination, penalty_intra_car)
    invt, waitt = Assignment.Skim_pt(G_start_pt, origin, destination)
    
    # Nested logit demand function
    def Demand_nestlogit_ownership(car_time, car_cost, car_park, inv_time, wait_time, 
                                   pt_price, dist, parameter, pop, emp, own):
        # Parameters
        mu = parameter[8]
        
        pop = np.asmatrix(pop).transpose()
        own = np.asmatrix(own).transpose()
        emp = np.asmatrix(emp)
        
        car_time = np.asmatrix(car_time)
        car_cost = np.asmatrix(car_cost)
        inv_time = np.asmatrix(inv_time)
        wait_time = np.asmatrix(wait_time)
        dist = np.asmatrix(dist)
        
        # Utility functions
        Vcar = parameter[0] + (parameter[1] * car_time) + (parameter[2] * (car_cost + car_park))
        Vpt = (parameter[3] * inv_time) + (parameter[4] * wait_time) + (parameter[5] * pt_price)
        Vslow = parameter[6] + (parameter[7] * dist)
        Vj = parameter[9] * np.log(emp)
        
        # Clip utility values to prevent overflow in exp()
        # Utilities larger than 700 would cause overflow
        MAX_UTILITY = 500
        Vcar = np.clip(Vcar, -MAX_UTILITY, MAX_UTILITY)
        Vpt = np.clip(Vpt, -MAX_UTILITY, MAX_UTILITY)
        Vslow = np.clip(Vslow, -MAX_UTILITY, MAX_UTILITY)
        
        # Car owners - three modes available
        exp_car = np.exp(Vcar / mu)
        exp_pt = np.exp(Vpt / mu)
        exp_slow = np.exp(Vslow / mu)
        denom_own = exp_car + exp_pt + exp_slow
        Lj_own = np.log(denom_own + 1e-10)
        
        Pcar_own = exp_car / denom_own
        Ppt_own = exp_pt / denom_own
        Pslow_own = exp_slow / denom_own
        
        V_dest_own = Vj + (mu * Lj_own)
        exp_dest_own = np.exp(V_dest_own)
        sum_dest_own = np.sum(exp_dest_own, axis=1)
        Pj_own = exp_dest_own / sum_dest_own
        
        Pcar_j_own = np.multiply(Pj_own, Pcar_own)
        Ppt_j_own = np.multiply(Pj_own, Ppt_own)
        Pslow_j_own = np.multiply(Pj_own, Pslow_own)
        
        pop_own = np.multiply(pop, own)
        Vol_car_own = np.multiply(pop_own, Pcar_j_own)
        Vol_pt_own = np.multiply(pop_own, Ppt_j_own)
        Vol_slow_own = np.multiply(pop_own, Pslow_j_own)
        
        EU_own = np.multiply(pop_own, np.log(sum_dest_own))
        
        # Non-car owners - two modes available
        denom_nocar = exp_pt + exp_slow
        Ij_nocar = np.log(denom_nocar + 1e-10)
        
        Ppt_nocar = exp_pt / denom_nocar
        Pslow_nocar = exp_slow / denom_nocar
        
        V_dest_nocar = Vj + (mu * Ij_nocar)
        exp_dest_nocar = np.exp(V_dest_nocar)
        sum_dest_nocar = np.sum(exp_dest_nocar, axis=1)
        Pj_nocar = exp_dest_nocar / sum_dest_nocar
        
        Ppt_j_nocar = np.multiply(Pj_nocar, Ppt_nocar)
        Pslow_j_nocar = np.multiply(Pj_nocar, Pslow_nocar)
        
        pop_nocar = np.multiply(pop, (1 - own))
        Vol_pt_nocar = np.multiply(pop_nocar, Ppt_j_nocar)
        Vol_slow_nocar = np.multiply(pop_nocar, Pslow_j_nocar)
        
        EU_nocar = np.multiply(pop_nocar, np.log(sum_dest_nocar))
        
        # Total volumes
        Vol_car = Vol_car_own
        Vol_pt = Vol_pt_own + Vol_pt_nocar
        Vol_slow = Vol_slow_own + Vol_slow_nocar
        EU = EU_own + EU_nocar
        
        return Vol_car, Vol_pt, Vol_slow, EU
    
    # Equilibrium iteration
    print(f"\nStarting equilibrium iterations (max {maxiter}, convergence threshold {maxdiff})...")
    car_time = car_time_0
    car_cost = car_cost_0
    
    for iter in range(maxiter):
        car_time_old = car_time.copy()
        car_cost_old = car_cost.copy()
        
        # Demand calculation
        v_car, v_pt, v_slow, EU = Demand_nestlogit_ownership(
            car_time, car_cost, carparking, invt, waitt,
            transitprice, dist, param,
            np.array(landuse['pop']),
            np.array(landuse['emp']),
            np.array(landuse['car_ownership'])
        )
        
        # Traffic assignment
        G_next_car = Assignment.RouteAssignment(
            np.asarray(v_car), G_car, origin, destination,
            vot_car, cost_km, penalty_intra_car
        )
        G_next_pt = Assignment.TransitAssignment(
            np.asarray(v_pt), G_pt, origin, destination,
            vot_in, vot_wait
        )
        
        # New travel times and costs
        car_time_new, car_cost_new, dist = Assignment.Skim_car(
            G_next_car, origin, destination, penalty_intra_car
        )
        
        # MSA smoothing
        car_time = car_time_new * smoothing + car_time_old * (1 - smoothing)
        car_cost = car_cost_new * smoothing + car_cost_old * (1 - smoothing)
        
        # Convergence check
        diff = np.mean(np.abs(car_time - car_time_old))
        print(f"Iteration {iter+1:2d}: Car demand = {np.sum(v_car):,.0f}, "
              f"PT demand = {np.sum(v_pt):,.0f}, Convergence = {diff:.4f}")
        
        if diff < maxdiff or iter >= maxiter - 1:
            print(f"\nConverged at iteration {iter+1}!")
            break
    
    # Store results
    results = {
        'scenario_name': scenario_name,
        'v_car': v_car,
        'v_pt': v_pt,
        'v_slow': v_slow,
        'EU': EU,
        'dist': dist,
        'G_car': G_next_car,
        'G_pt': G_next_pt,
        'iterations': iter + 1,
        'convergence': diff
    }
    
    return results


def compare_scenarios(baseline_results, policy_results, carparking, transitprice):
    """
    Compare baseline and policy scenarios and display the improvements.
    """
    print(f"\n{'='*80}")
    print("SCENARIO COMPARISON: Baseline vs Policy")
    print(f"{'='*80}\n")
    
    # Extract results
    v_car_base = baseline_results['v_car']
    v_pt_base = baseline_results['v_pt']
    v_slow_base = baseline_results['v_slow']
    EU_base = baseline_results['EU']
    dist_base = baseline_results['dist']
    
    v_car_policy = policy_results['v_car']
    v_pt_policy = policy_results['v_pt']
    v_slow_policy = policy_results['v_slow']
    EU_policy = policy_results['EU']
    dist_policy = policy_results['dist']
    
    # Calculate metrics for both scenarios
    print("BASELINE SCENARIO:")
    print("-" * 80)
    effect_base = Effect(v_car_base, v_pt_base, v_slow_base, dist_base, carparking, transitprice, EU_base)
    print(effect_base.to_string())
    
    print("\n\nPOLICY SCENARIO (Reduced Transit Waiting Times):")
    print("-" * 80)
    effect_policy = Effect(v_car_policy, v_pt_policy, v_slow_policy, dist_policy, carparking, transitprice, EU_policy)
    print(effect_policy.to_string())
    
    # Calculate changes
    print("\n\nIMPROVEMENTS (Policy vs Baseline):")
    print("=" * 80)
    
    # Trip totals
    car_trips_base = np.sum(v_car_base)
    pt_trips_base = np.sum(v_pt_base)
    slow_trips_base = np.sum(v_slow_base)
    total_base = car_trips_base + pt_trips_base + slow_trips_base
    
    car_trips_policy = np.sum(v_car_policy)
    pt_trips_policy = np.sum(v_pt_policy)
    slow_trips_policy = np.sum(v_slow_policy)
    total_policy = car_trips_policy + pt_trips_policy + slow_trips_policy
    
    # Mode shares
    car_share_base = (car_trips_base / total_base * 100) if total_base > 0 else 0
    pt_share_base = (pt_trips_base / total_base * 100) if total_base > 0 else 0
    slow_share_base = (slow_trips_base / total_base * 100) if total_base > 0 else 0
    
    car_share_policy = (car_trips_policy / total_policy * 100) if total_policy > 0 else 0
    pt_share_policy = (pt_trips_policy / total_policy * 100) if total_policy > 0 else 0
    slow_share_policy = (slow_trips_policy / total_policy * 100) if total_policy > 0 else 0
    
    # Expected Utility
    EU_base_total = np.sum(EU_base)
    EU_policy_total = np.sum(EU_policy)
    EU_change = EU_policy_total - EU_base_total
    EU_change_pct = (EU_change / abs(EU_base_total) * 100) if EU_base_total != 0 else 0
    
    # VKT and externalities
    VKT_base = np.sum(dist_base * v_car_base) / 1000
    VKT_policy = np.sum(dist_policy * v_car_policy) / 1000
    VKT_reduction = VKT_base - VKT_policy
    VKT_reduction_pct = (VKT_reduction / VKT_base * 100) if VKT_base > 0 else 0
    
    # Externality costs - coefficients from Effect.py
    # These coefficients represent cost per vehicle-km in euros
    WAITING_TIME_COST_COEF = 0.01    # €/VKT
    ACCIDENT_COST_COEF = 0.25        # €/VKT
    NOISE_COST_COEF = 0.081          # €/VKT
    EMISSION_COST_COEF = 0.004       # €/VKT
    CO2_COST_COEF = 3.15 * 0.085 * 1.5  # €/VKT (CO2 price * fuel consumption * factor)
    
    def calc_externalities(VKT):
        wt = WAITING_TIME_COST_COEF * VKT
        acc = ACCIDENT_COST_COEF * VKT
        noise = NOISE_COST_COEF * VKT
        emission = EMISSION_COST_COEF * VKT
        co2 = CO2_COST_COEF * VKT
        return wt + acc + noise + emission + co2
    
    ext_base = calc_externalities(VKT_base * 1000)
    ext_policy = calc_externalities(VKT_policy * 1000)
    ext_reduction = ext_base - ext_policy
    ext_reduction_pct = (ext_reduction / ext_base * 100) if ext_base > 0 else 0
    
    # Create comparison table
    comparison_data = {
        "Metric": [
            "Transit Trips",
            "Transit Mode Share (%)",
            "Car Trips",
            "Car Mode Share (%)",
            "Expected Utility",
            "Distance Travelled (1000 km)",
            "Total Externalities (€)"
        ],
        "Baseline": [
            f"{int(pt_trips_base):,}",
            f"{pt_share_base:.2f}%",
            f"{int(car_trips_base):,}",
            f"{car_share_base:.2f}%",
            f"{int(EU_base_total):,}",
            f"{VKT_base:.0f}",
            f"{int(ext_base):,}"
        ],
        "Policy": [
            f"{int(pt_trips_policy):,}",
            f"{pt_share_policy:.2f}%",
            f"{int(car_trips_policy):,}",
            f"{car_share_policy:.2f}%",
            f"{int(EU_policy_total):,}",
            f"{VKT_policy:.0f}",
            f"{int(ext_policy):,}"
        ],
        "Change": [
            f"{int(pt_trips_policy - pt_trips_base):+,}",
            f"{pt_share_policy - pt_share_base:+.2f}pp",
            f"{int(car_trips_policy - car_trips_base):+,}",
            f"{car_share_policy - car_share_base:+.2f}pp",
            f"{int(EU_change):+,} ({EU_change_pct:+.2f}%)",
            f"{-VKT_reduction:+.0f} ({-VKT_reduction_pct:+.1f}%)",
            f"{int(-ext_reduction):+,} ({-ext_reduction_pct:+.1f}%)"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Summary of key improvements
    print("\n\nKEY IMPROVEMENTS:")
    print("=" * 80)
    improvements = []
    
    if pt_trips_policy > pt_trips_base:
        improvements.append(f"✓ Public Transit Trips increased by {int(pt_trips_policy - pt_trips_base):,} "
                          f"({((pt_trips_policy - pt_trips_base)/pt_trips_base*100):.1f}%)")
    
    if pt_share_policy > pt_share_base:
        improvements.append(f"✓ Transit Mode Share increased by {pt_share_policy - pt_share_base:.2f} percentage points")
    
    if EU_policy_total > EU_base_total:
        improvements.append(f"✓ Expected Utility increased by {int(EU_change):,} ({EU_change_pct:.2f}%)")
    
    if VKT_policy < VKT_base:
        improvements.append(f"✓ Vehicle Kilometers Traveled reduced by {VKT_reduction:.0f} thousand km "
                          f"({VKT_reduction_pct:.1f}%)")
    
    if ext_policy < ext_base:
        improvements.append(f"✓ Total Externalities reduced by €{int(ext_reduction):,} ({ext_reduction_pct:.1f}%)")
    
    if car_trips_policy < car_trips_base:
        improvements.append(f"✓ Car Trips reduced by {int(car_trips_base - car_trips_policy):,} "
                          f"({((car_trips_base - car_trips_policy)/car_trips_base*100):.1f}%)")
    
    for improvement in improvements:
        print(improvement)
    
    if not improvements:
        print("⚠ No improvements detected. Consider adjusting the policy parameters.")
    
    print("=" * 80)
    
    return comparison_df


def main():
    """
    Main function to run baseline and policy scenarios and compare results.
    """
    print("="*80)
    print("TRANSIT WAITING TIME POLICY ANALYSIS")
    print("="*80)
    
    # Model parameters
    fuel_cost = 12
    fuel_con = 0.085
    cost_km = fuel_cost * fuel_con
    carparking = 10
    transitprice = 30
    penalty_intra_car = 30
    
    # Behavioral parameters
    alpha = 0.5
    beta_car = -0.08
    gamma_car = -0.05
    beta_inv = -0.05
    beta_wait = -0.08
    gamma_pt = -0.05
    alpha_slow = 0.1
    phi_dist = -0.5
    mu = 0.5
    theta = 1
    param = [alpha, beta_car, gamma_car, beta_inv, beta_wait, gamma_pt, 
             alpha_slow, phi_dist, mu, theta]
    
    vot_car = beta_car / gamma_car
    vot_in = beta_inv / gamma_pt
    vot_wait = beta_wait / gamma_pt
    
    # Car ownership parameters
    constant = 0.2
    income = 0.003
    dummy = -0.5
    param_carown = [constant, income, dummy]
    
    # Load landuse and zones
    print("\nLoading land use and zone data...")
    landuse, zones = Landuse.get_landuse_and_zones(param_carown)
    
    # Create base networks
    print("Creating transportation networks...")
    G_car_base = Network.RoadNetwork(zones.transpose())
    G_pt_base = Network.TransitNetwork(zones.transpose())
    
    # Add intra-zonal links
    G_car_base, G_pt_base = Assignment.add_intra_zonal_links(
        G_car_base, G_pt_base, zones=zones, landuse=landuse,
        vot_car=vot_car, vot_pt=vot_in, cost_km=cost_km,
        transitprice=transitprice, penalty_intra_car=penalty_intra_car
    )
    
    # Run baseline scenario
    baseline_results = run_scenario(
        "BASELINE",
        copy.deepcopy(G_car_base),
        copy.deepcopy(G_pt_base),
        landuse, zones, param, param_carown,
        vot_car, vot_in, vot_wait, cost_km, carparking, transitprice,
        penalty_intra_car, maxiter=50, maxdiff=0.01, smoothing=0.2
    )
    
    # Define policy: Reduce waiting times on high-frequency T-bana corridors
    print("\n" + "="*80)
    print("APPLYING TRANSIT POLICY")
    print("="*80)
    print("\nPolicy: Reduce waiting times by 50% on major T-bana corridors")
    print("This simulates improved service frequency on key transit routes.\n")
    print("Modified links:")
    
    policy_config = {
        # Major T-bana connections - reduce by 50%
        ('centerN', 'centerE'): 0.5,  # Centralen-Östermalm
        ('centerE', 'centerN'): 0.5,
        ('centerN', 'centerS'): 0.5,  # Centralen-Södermalm
        ('centerS', 'centerN'): 0.5,
        ('centerW', 'centerN'): 0.5,  # Kungsholm-Centralen
        ('centerN', 'centerW'): 0.5,
        ('centerN', 'N'): 0.5,        # Centralen-North (via centerN or centerW)
        ('N', 'centerN'): 0.5,
        ('centerW', 'N'): 0.5,
        ('N', 'centerW'): 0.5,
        # Additional key corridors - reduce by 40%
        ('NE', 'centerE'): 0.6,       # Danderyd-Östermalm
        ('centerE', 'NE'): 0.6,
        ('centerS', 'S'): 0.6,        # Södermalm-South
        ('S', 'centerS'): 0.6,
        ('centerS', 'SW'): 0.6,       # Södermalm-Southwest
        ('SW', 'centerS'): 0.6,
    }
    
    G_pt_policy = apply_transit_policy(copy.deepcopy(G_pt_base), policy_config)
    
    # Run policy scenario
    policy_results = run_scenario(
        "POLICY (Reduced Transit Waiting Times)",
        copy.deepcopy(G_car_base),
        G_pt_policy,
        landuse, zones, param, param_carown,
        vot_car, vot_in, vot_wait, cost_km, carparking, transitprice,
        penalty_intra_car, maxiter=50, maxdiff=0.01, smoothing=0.2
    )
    
    # Compare scenarios
    comparison_df = compare_scenarios(baseline_results, policy_results, carparking, transitprice)
    
    # Save results to CSV
    output_file = "policy_comparison_results.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return baseline_results, policy_results, comparison_df


if __name__ == "__main__":
    baseline_results, policy_results, comparison_df = main()
