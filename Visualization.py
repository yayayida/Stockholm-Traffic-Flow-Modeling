import pandas
import geopandas
import numpy as np
import folium
from folium.features import DivIcon
def Visualize(G_car, G_pt, zones, title=None, show_text_threshold=None):
    """
    Function plots the chloropeth map with zones as the base and lines from the network graph. 
    The width of the line is controled using the "Volume" in the traffic.
    """
    
    # Initilize the map using zones. The Geometry will be used to draw zones. 
    m = zones.explore(tiles="CartoDB positron", # use "CartoDB positron" tiles
                      cmap="Set1", # use "Set1" matplotlib colormap
                      tooltip=False,
                      style_kwds=dict(color="black") # use black outline
                    )
    
    # Title to map
    title_html = '''
                <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                '''.format(title)
    m.get_root().html.add_child(folium.Element(title_html))

    # Plotting lines for Car Traffic using car_network_graph
    carFg = folium.FeatureGroup("Car Traffic")
    for zone in G_car.edges:
        volume = G_car.edges[zone]['volume']
        p1 = zones.loc[zone[0], 'centroid']
        p2 = zones.loc[zone[1], 'centroid']
        coordinates = [p1, p2]
        folium.PolyLine(locations=coordinates,
                        tooltip=f"Car Volume:{volume:.3f}", 
                        weight=volume/5000,
                        opacity=0.5,
                        color = 'red').add_to(carFg)
        if show_text_threshold and volume > show_text_threshold:
            mid_lat = (p1[0] + p2[0]) / 2
            mid_lon = (p1[1] + p2[1]) / 2
            
            folium.map.Marker(
                [mid_lat, mid_lon],
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 10pt; font-weight: bold; color: darkred; background-color: rgba(255,255,255,0.6); padding: 2px; border-radius: 3px; display: inline-block;">{int(volume)}</div>'
                )
            ).add_to(carFg)
    carFg.add_to(m)

    # Plotting lines for Pubic Transit using pt_network_graph
    ptFg = folium.FeatureGroup("Public Transport Traffic")
    for zone in G_pt.edges:
        volume = G_pt.edges[zone]['volume']
        p1 = zones.loc[zone[0], 'centroid']
        p2 = zones.loc[zone[1], 'centroid']
        coordinates = [p1, p2]
        folium.PolyLine(locations=coordinates,
                        tooltip=f"PT Volume:{volume:.3f}", 
                        weight=volume/5000,
                        opacity=0.5,
                        color = 'green').add_to(ptFg)
        if show_text_threshold and volume > show_text_threshold:
            mid_lat = (p1[0] + p2[0]) / 2
            mid_lon = (p1[1] + p2[1]) / 2
            
            folium.map.Marker(
                [mid_lat, mid_lon],
                icon=DivIcon(
                    icon_size=(180,50),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 10pt; font-weight: bold; color: darkgreen; background-color: rgba(255,255,255,0.6); padding: 2px; border-radius: 3px; display: inline-block;">{int(volume)}</div>'
                )
            ).add_to(ptFg)
    ptFg.add_to(m)

    # Adding layer control on the map
    folium.LayerControl(position='topright').add_to(m)

    return m
