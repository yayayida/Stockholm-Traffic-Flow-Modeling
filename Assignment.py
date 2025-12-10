import pandas
import geopandas
from haversine import haversine
import networkx as nx
import numpy as np
import time
from tabulate import tabulate


def vdf(s, e, edge):
    tt = edge['distance'] * (1 + 0.09 * (edge['volume'] / edge['cap'])**2)
    return tt

def vdfInt(edge, vol):
    tt = edge['distance'] * (vol + 0.03 * vol ** 3 / edge['cap'] ** 2)
    return tt

def AllOrNothing(demand: np.ndarray, G: nx.DiGraph, origs: dict, dests: dict):
    AoN = {}
    for i in G.nodes():
        if i not in origs:  # skip if node not in demand matrix
            continue
        try:
            paths = nx.shortest_path(G, source=i, weight='weight', method="dijkstra")
        except:
            continue

        for j in paths:
            if j not in dests:
                continue

            o = origs[i]   # row index in demand matrix
            d = dests[j]   # column index

            demandx = demand[o, d]

            # Handle intra-zonal demand
            if i == j:
                if G.has_edge(i, i):
                    AoN[(i, i)] = AoN.get((i, i), 0) + demandx
                continue

            path = tuple(paths[j])
            pairs = [path[k: k + 2] for k in range(len(path) - 1)]
            for pair in pairs:
                AoN[pair] = AoN.get(pair, 0) + demandx

    return AoN


def FindLambda(G, x, x_aon):
    w_lo = 0
    w_hi = 1

    while w_hi - w_lo > 0.001:
        w = (w_lo + w_hi) / 2
        x_hat = w * x_aon + (1-w) * x
        t_wsum = 0
        l = 0
        t_l = np.zeros(len(x))
        for s,t in G.edges:
            t_l[l] = vdfInt(G.edges[s,t], x_hat[l])
            t_wsum += t_l[l] * (x_aon[l] - x[l])
            l += 1

        if t_wsum > 0:
            w_hi = w
        else:
            w_lo = w

    return w
def Skim_car(G, origs, dests, penalty_intra_car):   
    nz = len(origs)
    t_ij = np.zeros((nz, nz))
    c_ij = np.zeros((nz, nz))
    d_ij = np.zeros((nz, nz))
    sp_time = 0
    skim_time = 0
    for i in origs.keys():
        start_sp = time.time()
        #paths = nx.single_source_dijkstra_path(G, source=i, weight='weight')
        paths = nx.shortest_path(G, source=i, weight='weight', method="dijkstra")
        sp_time += time.time() - start_sp
        #print(f"AoN {i}")
        start_skim = time.time()
        for j in paths:
            if j in dests:
                o = origs[i]
                d = dests[j]
                # Handle intra-zonal trips using (i, i) edge directly
                if i == j and G.has_edge(i, i):
                    t_ij[o, d] = G.edges[i, i]['traveltime']
                    c_ij[o, d] = G.edges[i, i]['cost'] + penalty_intra_car
                    d_ij[o, d] = G.edges[i, i]['distance']
                    continue
                path = tuple(paths[j])
                pairs = [path[i: i + 2] for i in range(len(path)-1)]
                for pair in pairs:
                    (s,t) = pair
                    t_ij[o,d] += G.edges[s,t]['traveltime']
                    c_ij[o,d] += G.edges[s,t]['cost']
                    d_ij[o,d] += G.edges[s,t]['distance']
        skim_time += time.time() - start_skim
    return (np.asmatrix(t_ij),np.asmatrix(c_ij), np.asmatrix(d_ij))


def Skim_pt(G, origs, dests):
    # Fixed traveltimes on the diagonal
    nz = len(origs)
    #t_ij = np.eye(nz) * 2
    #tw_ij = np.eye(nz) * 2
    t_ij = np.zeros((nz, nz))
    tw_ij = np.zeros((nz, nz))
    sp_time = 0
    skim_time = 0
    for i in origs.keys():
        start_sp = time.time()
        paths = nx.single_source_dijkstra_path(G, source=i, weight='weight')
        sp_time += time.time() - start_sp
        #print(f"AoN {i}")
        start_skim = time.time()
        for j in paths:
            if j in dests:
                o = origs[i]
                d = dests[j]
                if i == j and G.has_edge(i, i):
                    o = origs[i]
                    d = dests[j]
                    t_ij[o, d] = G.edges[i, i]['inv_time']
                    tw_ij[o, d] = G.edges[i, i]['wait_time']
                    continue
                path = tuple(paths[j])
                pairs = [path[i: i + 2] for i in range(len(path)-1)]
                for pair in pairs:
                    (s,t) = pair
                    t_ij[o,d] += G.edges[s,t]['inv_time']
                    tw_ij[o,d] += G.edges[s,t]['wait_time']
        skim_time += time.time() - start_skim
    return (np.asmatrix(t_ij),np.asmatrix(tw_ij))

def RouteAssignment(demand, G_base, orig, dest, VOT, cost_per_km,penalty_intra_car):
    gap = 0.0001
    iters = 10
    G = G_base.copy()
    for s, t in G.edges:
        if s == t:
            #continue   # Keep intra-zonal cost fixed, avoid update
            G.edges[s, t]['weight'] = (
            VOT * G.edges[s, t]['traveltime']
            + G.edges[s, t]['cost']
            + penalty_intra_car
            )
        else:
            ttime = vdf(s, t, G.edges[s, t])
            G.edges[s, t]['cost'] = G.edges[s, t]['distance'] * cost_per_km
            G.edges[s, t]['traveltime'] = ttime
            G.edges[s, t]['weight'] = ttime * VOT + G.edges[s, t]['cost']
    
    aon = AllOrNothing(demand, G, orig, dest)
    # set the link flows in the graph
    for s, t in G.edges:
        if (s, t) in aon:
            x = aon[(s, t)]
        else:
            x = 0
        G.edges[s, t]['volume'] = x
    sumT = sum(sum(demand))

    diff = 100
    relgap = 1
    k=1

    table = list()
    print("ROUTE ASSIGNMENT")
    print("---------------------------------------------------------------------------------------------")
    table.append(['ITR', 'Total System Travel Time','Shortest Path Travel Times','Relgap','Changes in in volume'])
    while relgap > gap and k < iters:

        # some vectors to keep data in
        diffs = np.zeros(len(G.edges))
        x = np.zeros(len(G.edges))
        x_aon = np.zeros(len(G.edges))
        times = np.zeros(len(G.edges))
        # update weights for the current solution
        l = 0
        for s, t in G.edges:
            if s == t:
        # Keep intra-zonal cost fixed, avoid update
                #continue
                times[l] = G.edges[s, t]['traveltime']
                G.edges[s, t]['weight'] = (
                    VOT * times[l]
                    + G.edges[s, t]['cost']
                    + penalty_intra_car
                )
            else:
                times[l] = vdf(s, t, G.edges[s, t])
                G.edges[s, t]['traveltime'] = times[l]
                G.edges[s, t]['weight'] = times[l] * VOT + G.edges[s, t]['cost']
            # save the current solution in a vector
            x[l] = G.edges[s, t]['volume']
            l += 1

        # assign all flow on shortest paths
        aon = AllOrNothing(demand, G, orig, dest)

        l = 0
        for s, t in G.edges:
            # put the all-or-nothing solution in a vector
            if (s, t) in aon:
                x_aon[l] = aon[(s, t)]
            l += 1

        l = 0
        # choose lambda (step size)
        w = FindLambda(G, x, x_aon)
        for s, t in G.edges:
            # update the current solution by stepping towards x*
            G.edges[s, t]['volume'] = (1 - w) * x[l] + w * x_aon[l]
            l += 1

        # Total System Travel Time
        TSTT = sum(x * times)

        # Shortest Path Travel Times
        # i.e. current travel times, shortest path (aon) flows
        SPTT = sum(x_aon * times)
        diffs = x - x_aon
        diff = np.linalg.norm(diffs, 1)
        relgap = (TSTT / SPTT) - 1
        AEC = (TSTT - SPTT) / sumT

        # Print output as table
        table.append([k, TSTT,SPTT,relgap,diff])
        k += 1
    print(tabulate(table, headers='firstrow'))
    print("---------------------------------------------------------------------------------------------")
    print(f"assignment: k = {k} relgap = {relgap:.6f}, AEC = {AEC:.6f}")
    return G

def TransitAssignment(demand, G, orig, dest,VOT_in, VOT_wait):
  for s, t in G.edges:
      G.edges[s, t]['weight'] = G.edges[s, t]['inv_time'] * VOT_in + G.edges[s, t]['wait_time'] * VOT_wait
  aon = AllOrNothing(demand, G, orig, dest)
  # set the link flows in the graph
  for s, t in G.edges:
      if (s, t) in aon:
          x = aon[(s, t)]
      else:
          x = 0
      G.edges[s, t]['volume'] = x
  return (G)

def add_intra_zonal_links(G_car, G_pt, zones, landuse, vot_car, vot_pt, cost_km, transitprice, penalty_intra_car):
    """
    Adds intra-zonal links to car and transit networks with estimated travel times and costs.

    Parameters:
        G_car (nx.Graph): car network
        G_pt (nx.Graph): transit network
        zones (list): zone names
        landuse (DataFrame): includes zone area under key 'area_2'
        vot_car (float): value of time (car) in SEK/min
        vot_pt (float): value of time (pt) in SEK/min
        cost_km (float): car cost per km
        carparking (float): car parking cost per trip
        transitprice (float): fixed cost for PT trip

    Returns:
        Updates G_car and G_pt in-place
    """
    #true_zones = landuse['area'].tolist()
    #zone_to_area = dict(zip(true_zones, landuse['area_2']))
    #for zone in true_zones:
    for zone, area_m2 in zip(landuse['area'], landuse['area_2']):
        #area = zone_to_area[zone]  # in m²
        diagonal_dist_km = np.sqrt(area_m2) / 1000  # km

        # Synthetic travel times
        time_car_min = (diagonal_dist_km / 30) * 60  # 30 km/h
        time_pt_min = (diagonal_dist_km / 20) * 60 + 5  # 20 km/h + wait time

        # Costs
        cost_car = diagonal_dist_km * cost_km 
        cost_pt = transitprice

        # Generalized cost
        gc_car = vot_car * time_car_min + cost_car + penalty_intra_car
        gc_pt = vot_pt * time_pt_min + cost_pt

        # Add dummy edge to car network
        G_car.add_edge(zone, zone,
                       distance=diagonal_dist_km,
                       traveltime=time_car_min,
                       cost=cost_car,
                       weight=gc_car,
                       volume=0,
                       cap=99999)

        # Add dummy edge to pt network
        G_pt.add_edge(zone, zone,
                      inv_time=time_pt_min,
                      wait_time=5,
                      volume=0,
                      weight=gc_pt)

    return G_car, G_pt


