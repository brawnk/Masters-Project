#import matplotlib as mpl          # Plotting library
#mpl.use('Agg')                    # Lets us save without drawing the picture. Needed for running on epp cluster. Comment it out if you want to just display the plots, and not save.
from matplotlib import cm         # Colour maps
import matplotlib.pyplot as plt   # Lets us plot
import pandas as pd               # Reads in data from csv
import numpy as np                # Does all the maths
#import os                         # Needed to read files on os
#import sys                        # Needed to get command line options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

"""Return two numbers in a string rather than adding i.e combine_nbr(1,2) = 12"""
def combine_nbr(numa, numb):
    return int(str(numa)+str(numb))

"""Return array containing all numbers in the input string separated if there are nondigits inbetween i.e. extract_nbr("C:0 T:9 P:1 W:430") = [0,9,1,430]"""
def extract_nbr(input_str):
    out_number = []
    sect = 0
    nbr = False
    for ele in input_str:
        if ele.isdigit():
            nbr = True
            sect=combine_nbr(sect,ele)
        else:
            if nbr == True:
                out_number.append(sect)
                sect = 0
            nbr = False
    out_number.append(sect)
    return out_number
 
"""Return appends arr2 to array 1 in a single array i.e. join_arr([0,1],[2,3,4]) = [0,1,2,3,4]"""
def join_arr(arr1, arr2):
    fin = []
    for i in range(0,len(arr1)):
        fin.append(arr1[i])
    for j in range(0,len(arr2)):
        fin.append(arr2[j])
    return fin

"""Separate WireID column in col variable into separate columns for C, T, P & W"""
def CTPW_split(df,col):
    C = []
    T = []
    P = []
    W = []

    for i in range(0,len(df)):
        curr_arr = extract_nbr(df.iloc[i,col])
        C.append(curr_arr[0])
        T.append(curr_arr[1])
        P.append(curr_arr[2])
        W.append(curr_arr[3])

    df.loc[:,'Cryo']= C
    df.loc[:,'TPC']= T
    df.loc[:,'Plane']= P
    df.loc[:,'Wirenum']= W

"""Correction for plane 2"""
def Plane_2_Corr(df):
    df['Plane Corrected'] = np.nan
    planecorr_ind = df.columns.get_loc("Plane Corrected")
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")
    
    for i in range(0,len(df)):
        tpc = df.iloc[i,tpc_ind]
        if tpc in [8,9,10,11]:
            df.iloc[i,planecorr_ind] = 960 + df.iloc[i,wirenum_ind]
        elif tpc in [4,5,6,7]:
            df.iloc[i,planecorr_ind] = 480 + df.iloc[i,wirenum_ind]
        elif tpc in [0,1,2,3]:
            df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind]
        else:
            print("Plane 2 correction failed!")
            exit(1)

"""Correction for plane 1"""
def Plane_1_Corr(df):
    df['Plane Corrected'] = np.nan
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")
    planecorr_ind = df.columns.get_loc("Plane Corrected")
    
    uni = df['TPC'].unique()
    sects = len(uni) #How many TPC's does the track enter
    uni.sort()
    if (sects > 3):
        print("Correction error too many TPC's!")
        exit(1)
    
    sect = [0, 0, 0]
    if (1 in uni) or (2 in uni):
        sect[0] = 1
    if (5 in uni) or (6 in uni):
        sect[1] = 1
    if (9 in uni) or (10 in uni):
        sect[2] = 1

    if (sects == 3): #Track goes in all 3 sections of the detector
        dfup = df.loc[df['TPC'] == uni[0]] #upstream tpc
        dfmid = df.loc[df['TPC'] == uni[1]] #midstream tpc
        dfdown = df.loc[df['TPC'] == uni[2]] #downstream tpc
        
        max910 = max(dfdown['Wirenum'])
        min56 = min(dfmid['Wirenum'])
        corrmiddown = max910-min56 + 1
        min12 = min(dfup['Wirenum'])
        max56 = max(dfmid['Wirenum']) + corrmiddown
        corrupmid = max56 - min12 + 1
        flippt = max910 + min12 + corrupmid
        
        for i in range(len(df)):
            tpc = df.iloc[i,tpc_ind]
            if (tpc in [1, 2]):
                df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind] + corrupmid)
            elif (tpc in [5, 6]):
                df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind] + corrmiddown)
            else:# (tpc in [9, 10]):
                df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind])
    
    elif ((sects == 1) or (sect == [1, 0, 1])): #track only in 1 section or in tpc12 and tpc910
        for i in range(len(df)):
            multi = 0
            tpc = df.iloc[i,tpc_ind]
            if (tpc in [9,10]):
                multi = 2
            elif tpc in [5,6]:
                multi = 1
            df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind]+1148*multi
    else: #(sects == 2):
        if (sect[0] == 1): #Track only in tpc12 and tpc56 Sect = [1,1,0]
            dfup = df.loc[df['TPC'] == uni[0]] #upstream tpc
            dfmid = df.loc[df['TPC'] == uni[1]] #midstream tpc
            max56 = max(dfmid['Wirenum'])
            min12 = min(dfup['Wirenum'])
            
            corrupmid = max56 - min12 + 1
            max12 = max(dfup['Wirenum']) + corrupmid
            min56 = min(dfmid['Wirenum'])
            flippt = max12 + min56
            
            for i in range(len(df)):
                tpc = df.iloc[i,tpc_ind]
                if (tpc in [1,2]):
                    df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind] + corrupmid)
                else:
                    df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind])
        else:# (sect[2] == 1): #Track only in tpc56 and tpc910 Sect = [0,1,1]
            dfmid = df.loc[df['TPC'] == uni[0]] #midstream tpc
            dfdown = df.loc[df['TPC'] == uni[1]] #downstream tpc
            min56 = min(dfmid['Wirenum'])  
            max910 = max(dfdown['Wirenum'])
            corrmiddown = max910-min56+1
            
            max56 = max(dfmid['Wirenum']) + corrmiddown
            min910 = min(dfdown['Wirenum'])
            flippt = max56 + min910
            for i in range(len(df)):
                tpc = df.iloc[i,tpc_ind]
                if (tpc in [9, 10]):
                    df.iloc[i,planecorr_ind] = flippt - df.iloc[i,wirenum_ind]
                else:
                    df.iloc[i,planecorr_ind] = flippt - (df.iloc[i,wirenum_ind] + corrmiddown)

"""Correction for plane 0"""
def Plane_0_Corr(df):
    df['Plane Corrected'] = np.nan
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")
    planecorr_ind = df.columns.get_loc("Plane Corrected")
    
    uni = df['TPC'].unique()
    sects = len(uni) #How many of the three sections does the track enter
    uni.sort()
    if (sects > 3):
        print("Correction error too many TPC's!")
        exit(1)
    
    sect = [0, 0, 0]
    if (1 in uni) or (2 in uni):
        sect[0] = 1
    if (5 in uni) or (6 in uni):
        sect[1] = 1
    if (9 in uni) or (10 in uni):
        sect[2] = 1
    
    if (sects == 3): #Track goes in all 3 sections of the detector
        dfup = df.loc[df['TPC'] == uni[0]] #upstream tpc
        dfmid = df.loc[df['TPC'] == uni[1]] #midstream tpc
        dfdown = df.loc[df['TPC'] == uni[2]] #downstream tpc
        
        max12 = max(dfup['Wirenum'])
        min56 = min(dfmid['Wirenum'])
        corrupmid = max12-min56 + 1
        max56 = max(dfmid['Wirenum'])+corrupmid
        min910 = min(dfdown['Wirenum'])
        corrmiddown = max56-min910 + 1

        for i in range(len(df)):
            tpc = df.iloc[i,tpc_ind]
            if (tpc in [1, 2]):
                df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind]
            elif (tpc in [5, 6]):
                df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind] + corrupmid
            else:# (tpc in [9, 10]):
                df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind] + corrmiddown
    
    elif ((sects == 1) or (sect == [1, 0, 1])): #track only in 1 section or in tpc12 and tpc910
        for i in range(len(df)):
            multi = 0
            tpc = df.iloc[i,tpc_ind]
            if (tpc in [5,6]):
                multi = 1
            elif tpc in [9,10]:
                multi = 2
            df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind]+1148*multi
            
    else: #(sects == 2):
        if (sect[0] == 1): #Track only in tpc12 and tpc56
            dfup = df.loc[df['TPC'] == uni[0]] #upstream tpc
            dfmid = df.loc[df['TPC'] == uni[1]] #midstream tpc
            max12 = max(dfup['Wirenum'])
            min56 = min(dfmid['Wirenum'])
            corrupmid = max12-min56 + 1
            
            for i in range(len(df)):
                tpc = df.iloc[i,tpc_ind]
                if (tpc in [5, 6]):
                    df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind] + corrupmid
                else:
                    df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind]
                    
        else:# (sect[2] == 1): #Track only in tpc56 and tpc910
            dfmid = df.loc[df['TPC'] == uni[0]] #midstream tpc
            dfdown = df.loc[df['TPC'] == uni[1]] #downstream tpc
            max56 = max(dfmid['Wirenum'])  
            min910 = min(dfdown['Wirenum'])
            corrmiddown = max56-min910 + 1
            
            for i in range(len(df)):
                tpc = df.iloc[i,tpc_ind]
                if (tpc in [9, 10]):
                    df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind] + corrmiddown + 1148
                else:
                    df.iloc[i,planecorr_ind] = df.iloc[i,wirenum_ind] + 1148
                    
"""Combines plane corrections for all 3 planes into 1 function for easier readability"""
def Plane_Corr(df, plane):
    if (plane == 2):
        Plane_2_Corr(df)
    elif (plane == 1):
        Plane_1_Corr(df)
    else:
        Plane_0_Corr(df)
        
"""Extract particle name from the input filename - used for names of saved graphs"""
def name_extract(filename):
    
    prev = 1
    particle = ""
    for i in filename:
        if (prev == 1) & (i != "_"):
            particle += i
        else:
            prev = 0
            exit
    return(particle)  

"""Bin data (from incol) into numbins between binmin and binmax and return a new column which contains the halfway point of the bin that row is in."""
def BinData(df,incol,numbins, binmax, binmin):
    binwidth = (binmax-binmin)/numbins
    bins = []
    midpoints = []
    for i in range(0,numbins+1):
        bins.append(binmin)
        midpoints.append(binmin+binwidth/2.0)
        binmin += binwidth
    del midpoints[-1]

    return pd.cut(df[incol],bins,labels = midpoints, retbins=False)

def midind(df,i):
    planecorr_ind = df.columns.get_loc("Plane Corrected")
    return int(df.iloc[i,planecorr_ind])

def binind(binpoint, numbins):
    binwidth = 4500.0/numbins
    return int((binpoint/binwidth)-0.5)

def fillplot(df, wirerange, numbins):
    plotdf = df.loc[(df['Plane Corrected'] >= wirerange[0]) & (df['Plane Corrected'] < wirerange[1])]
    plotdata = np.zeros((numbins, numbins))

    for i in range(len(plotdf)):
        plotdata[binind(plotdf.iloc[i,8], numbins)][midind(plotdf,i)] = plotdf.iloc[i,0]
    return plotdata

def trackgraph(data, PlotLocation,size):
    xlen = size[0]/100.0
    ylen = size[1]/100.0
    fig = plt.figure(frameon=False)       # Remove the white space around the image. Ensures image only consists of data
    fig.set_size_inches(xlen,ylen)              # Set the size of the image, in inches
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # Ensures image only consists of data
    ax.set_axis_off()                     # Ensures image only consists of data
    fig.add_axes(ax)                      # Add the object to the figure
    ax.imshow(data,origin='lower',aspect='equal',cmap=cm.jet)  # Plot the numpy array, with the cm.jet colour map
    fig.savefig(PlotLocation,dpi=100)     # Save to the given location, at 100dpi. This gives us a 500x500 pixel image with the 5x5 inch size
    plt.close()

def Single_Track_Plot(df, plotrange, num_bins, Plot_Location):
    df['Binned Peak Time'] = BinData(df, 'PeakTime', numbins, 4500, 0)
    plotdata = fillplot(df, plotrange, num_bins)
    trackgraph(plotdata, Plot_Location)



""" Inputs by user """
#file = "debug.csv"               #USER INPUT
file = "kao_1000.csv"               #USER INPUT
saveloc = "test full event plot/"                  #USER INPUT
planes = [0,1,2]                        #USER INPUT
evtnum_start = 1                         #USER INPUT
evtnum_end = 1000                        #USER INPUT

""" Code starts here """



df = pd.read_csv(file)
print("file loaded")
CTPW_split(df,7)
print("split done")
particle = name_extract(file)

# 2) Go through the data on an event by event basis
# You'll need to sort the data into a sensible object.
# You can try to use a pandas.DataFrame object or (the way I do it) use a python dictionary.
# Try ordering you object in a logical manner, i.e. Each Plane has some TPCs, and each TPC has the hit and its properties.

timenumbins = 500 #time number of bins
count = 0
for evtnum in range(evtnum_start, evtnum_end + 1):
    if (evtnum % 10 == 0):
        print("running event number: %d" % evtnum)
    
    evtdf = df[df.evtNum == evtnum]
    
    if (len(evtdf) == 0):
        print("No event for evtnum %d!" % evtnum)
        continue
    
    for plane in planes:
        numbins = 3443 #track number of bins
        if (plane == 2):
            numbins = 1440
        
        planedf = evtdf[evtdf.Plane == plane]
        if (len(planedf) == 0):
            print("No plane data for evtnum %d plane %d!" % (evtnum, plane))
            continue
        
        tracks = planedf['TrackID'].unique()
        
        end = 0
        start = numbins
        red_plane_df = pd.DataFrame()
        evtlist = []
        
        for track in tracks:
            #print("evtnum: %d, track: %d, plane: %d" % (evtnum, track, plane))
            trackdf = planedf[planedf.TrackID == track]
            
            if (len(trackdf) == 0):
                #print("Track too short! evtnum: %d, track: %d, plane: %d" % (evtnum, track, plane))
                continue
            
            red_track_df = pd.DataFrame({'TrackID': trackdf.loc[:,'TrackID'].values,
                                         'PeakTime': BinData(trackdf, 'PeakTime', timenumbins, 4500, 0),
                                         'Integral': trackdf.loc[:,'Integral'].values,
                                         'PDG': trackdf.loc[:,'matchedTrueTrackPDG'].values,
                                         'TPC': trackdf.loc[:,'TPC'].values,
                                         'Plane': trackdf.loc[:,'Plane'].values,
                                         'Wirenum': trackdf.loc[:,'Wirenum'].values})

            Plane_Corr(red_track_df, plane)
            xmin = int(min(red_track_df['Plane Corrected']))
            xmax = int(max(red_track_df['Plane Corrected']))
            if (xmin < start):
                start = xmin
            if (xmax > end):
                end = xmax
                
            red_track_df['PeakTime'] = np.asarray(red_track_df['PeakTime'])
            evtlist.extend(red_track_df.values.tolist())

        evtlen = end-start
        plotdata = np.zeros((timenumbins, evtlen))
        
        red_plane_df = pd.DataFrame(evtlist, columns = ['Integral', 'PDG', 'PeakTime', 'Plane', 'TPC', 'TrackID', 'Wirenum', 'Plane Corrected'])
        #print(red_plane_df)
        if(red_plane_df['Plane Corrected'] < 0).any():
            print("plane corr negative! %d" % evtnum)
          
        for track in tracks:
            #print("evtnum: %d, track: %d, plane: %d" % (evtnum, track, plane))
            trackdf = red_plane_df[red_plane_df.TrackID == track]
            
            if len(trackdf) < 1:
                #print("track too short!")
                continue

            wiremin = min(trackdf['Plane Corrected']) #minimum value of wire
            wiremax = max(trackdf['Plane Corrected']) #maximum value of wire
            plotrange = (wiremin, wiremax)
            
            plotdf = trackdf.loc[(trackdf['Plane Corrected'] >= plotrange[0]) & (trackdf['Plane Corrected'] < plotrange[1])]
            peaktime_ind = plotdf.columns.get_loc("PeakTime")
            integral_ind = plotdf.columns.get_loc("Integral")
            for i in range(len(plotdf)):
                y_ind = binind(plotdf.iloc[i,peaktime_ind],timenumbins)
                x_ind = midind(plotdf,i)-start
                plotdata[y_ind][x_ind] += plotdf.iloc[i,integral_ind]
        #loc = "graphs/"+ str(particle) + "/Plane_" + str(plane) + "/" + str(particle) + "_event_num_" + str(evtnum) + "_Plane_" + str(plane) + "_" +str(max(tracks)) + "_Tracks"
        loc = saveloc + str(particle) + "_event_num_" + str(evtnum) + "_Plane_" + str(plane) + "_" +str(max(tracks)) + "_Tracks"
        trackgraph(plotdata, loc, [evtlen, timenumbins])
        count += 1
        
print("completed for %s produced %d images" % (particle, count))
