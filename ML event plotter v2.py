# import matplotlib as mpl
# mpl.use('Agg')  # Lets us save without drawing the picture. Needed for running on epp cluster. Comment it out if you want to just display the plots, and not save.
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import random as rand
# import os
# import sys
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)


def combine_nbr(numa, numb):
    """Return two numbers in a string rather than adding i.e combine_nbr(1,2) = 12"""
    return int(str(numa) + str(numb))


def extract_nbr(input_str):
    """Return array containing all numbers in the input string separated if there are non-digits inbetween i.e. extract_nbr("C:0 T:9 P:1 W:430") = [0,9,1,430]"""
    out_number = []
    sect = 0
    nbr = False
    for ele in input_str:
        if ele.isdigit():
            nbr = True
            sect = combine_nbr(sect, ele)
        else:
            if nbr is True:
                out_number.append(sect)
                sect = 0
            nbr = False
    out_number.append(sect)
    return out_number


def join_arr(arr1, arr2):
    """Return appends arr2 to array 1 in a single array i.e. join_arr([0,1],[2,3,4]) = [0,1,2,3,4]"""
    fin = []
    for i in range(0, len(arr1)):
        fin.append(arr1[i])
    for j in range(0, len(arr2)):
        fin.append(arr2[j])
    return fin


def CTPW_split(df, col):
    """Separate WireID column in col variable into separate columns for C, T, P & W"""
    C = []
    T = []
    P = []
    W = []

    for i in range(0, len(df)):
        curr_arr = extract_nbr(df.iloc[i, col])
        C.append(curr_arr[0])
        T.append(curr_arr[1])
        P.append(curr_arr[2])
        W.append(curr_arr[3])

    df.loc[:, 'Cryo'] = C
    df.loc[:, 'TPC'] = T
    df.loc[:, 'Plane'] = P
    df.loc[:, 'Wirenum'] = W


def Plane_2_Correction(df):
    """Correction for plane 2"""
    df['PlaneCorrected'] = np.nan
    planecorr_ind = df.columns.get_loc("PlaneCorrected")
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")

    for i in range(0, len(df)):
        tpc = df.iloc[i, tpc_ind]
        if tpc in [8, 9, 10, 11]:
            df.iloc[i, planecorr_ind] = 960 + df.iloc[i, wirenum_ind]
        elif tpc in [4, 5, 6, 7]:
            df.iloc[i, planecorr_ind] = 480 + df.iloc[i, wirenum_ind]
        elif tpc in [0, 1, 2, 3]:
            df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind]
        else:
            print("Plane 2 correction failed!")
            exit(1)


def Plane_1_Corr(df):
    """Correction for plane 1. Flip points along the vertical centre to match the directions of planes 0 and 2."""
    df['PlaneCorrected'] = np.nan
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")
    planecorr_ind = df.columns.get_loc("PlaneCorrected")

    uni = df['TPC'].unique()
    sects = len(uni)  # number of TPC's the track enters
    uni.sort()
    if sects > 3:
        print("Correction error too many TPC's!")
        exit(1)

    sect = [0, 0, 0]
    if (1 in uni) or (2 in uni):
        sect[0] = 1
    if (5 in uni) or (6 in uni):
        sect[1] = 1
    if (9 in uni) or (10 in uni):
        sect[2] = 1

    if sects == 3:  # Track goes in all 3 sections of the detector
        dfup = df.loc[df['TPC'] == uni[0]]  # upstream tpc
        dfmid = df.loc[df['TPC'] == uni[1]]  # midstream tpc
        dfdown = df.loc[df['TPC'] == uni[2]]  # downstream tpc

        max910 = max(dfdown['Wirenum'])
        min56 = min(dfmid['Wirenum'])
        corrmiddown = max910 - min56 + 1
        min12 = min(dfup['Wirenum'])
        max56 = max(dfmid['Wirenum']) + corrmiddown
        corrupmid = max56 - min12 + 1
        flippt = max910 + min12 + corrupmid

        for i in range(len(df)):
            tpc = df.iloc[i, tpc_ind]
            if tpc in [1, 2]:
                df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind] + corrupmid)
            elif tpc in [5, 6]:
                df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind] + corrmiddown)
            else:  # (tpc in [9, 10]):
                df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind])

    elif (sects == 1) or (sect == [1, 0, 1]):  # track only in 1 section or in tpc12 and tpc910
        for i in range(len(df)):
            multi = 0
            tpc = df.iloc[i, tpc_ind]
            if tpc in [9, 10]:
                multi = 2
            elif tpc in [5, 6]:
                multi = 1
            df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + 1148 * multi

    else:  # (sects == 2):
        if sect[0] == 1:  # Track only in tpc12 and tpc56 Sect = [1,1,0]
            dfup = df.loc[df['TPC'] == uni[0]]  # upstream tpc
            dfmid = df.loc[df['TPC'] == uni[1]]  # midstream tpc
            max56 = max(dfmid['Wirenum'])
            min12 = min(dfup['Wirenum'])

            corrupmid = max56 - min12 + 1
            max12 = max(dfup['Wirenum']) + corrupmid
            min56 = min(dfmid['Wirenum'])
            flippt = max12 + min56

            for i in range(len(df)):
                tpc = df.iloc[i, tpc_ind]
                if tpc in [1, 2]:
                    df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind] + corrupmid)
                else:
                    df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind])

        else:  # (sect[2] == 1): Track only in tpc56 and tpc910 Sect = [0,1,1]
            dfmid = df.loc[df['TPC'] == uni[0]]  # midstream tpc
            dfdown = df.loc[df['TPC'] == uni[1]]  # downstream tpc
            min56 = min(dfmid['Wirenum'])
            max910 = max(dfdown['Wirenum'])
            corrmiddown = max910 - min56 + 1

            max56 = max(dfmid['Wirenum']) + corrmiddown
            min910 = min(dfdown['Wirenum'])
            flippt = max56 + min910
            for i in range(len(df)):
                tpc = df.iloc[i, tpc_ind]
                if tpc in [9, 10]:
                    df.iloc[i, planecorr_ind] = flippt - df.iloc[i, wirenum_ind]
                else:
                    df.iloc[i, planecorr_ind] = flippt - (df.iloc[i, wirenum_ind] + corrmiddown)


def Plane_0_Corr(df):
    """Correction for plane 0"""
    df['PlaneCorrected'] = np.nan
    tpc_ind = df.columns.get_loc("TPC")
    wirenum_ind = df.columns.get_loc("Wirenum")
    planecorr_ind = df.columns.get_loc("PlaneCorrected")

    uni = df['TPC'].unique()
    sects = len(uni)  # Number of sections the track enters
    uni.sort()
    if sects > 3:
        print("Correction error too many TPC's!")
        exit(1)

    sect = [0, 0, 0]
    if (1 in uni) or (2 in uni):
        sect[0] = 1
    if (5 in uni) or (6 in uni):
        sect[1] = 1
    if (9 in uni) or (10 in uni):
        sect[2] = 1

    if sects == 3:  # Track goes in all 3 sections of the detector
        dfup = df.loc[df['TPC'] == uni[0]]  # upstream tpc
        dfmid = df.loc[df['TPC'] == uni[1]]  # midstream tpc
        dfdown = df.loc[df['TPC'] == uni[2]]  # downstream tpc

        max12 = max(dfup['Wirenum'])
        min56 = min(dfmid['Wirenum'])
        corrupmid = max12 - min56 + 1
        max56 = max(dfmid['Wirenum']) + corrupmid
        min910 = min(dfdown['Wirenum'])
        corrmiddown = max56 - min910 + 1

        for i in range(len(df)):
            tpc = df.iloc[i, tpc_ind]
            if tpc in [1, 2]:
                df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind]
            elif tpc in [5, 6]:
                df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + corrupmid
            else:  # (tpc in [9, 10]):
                df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + corrmiddown

    elif (sects == 1) or (sect == [1, 0, 1]):  # track only in 1 section or in tpc12 and tpc910
        for i in range(len(df)):
            multi = 0
            tpc = df.iloc[i, tpc_ind]
            if tpc in [5, 6]:
                multi = 1
            elif tpc in [9, 10]:
                multi = 2
            df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + 1148 * multi

    else:  # (sects == 2):
        if sect[0] == 1:  # Track only in tpc12 and tpc56
            dfup = df.loc[df['TPC'] == uni[0]]  # upstream tpc
            dfmid = df.loc[df['TPC'] == uni[1]]  # midstream tpc
            max12 = max(dfup['Wirenum'])
            min56 = min(dfmid['Wirenum'])
            corrupmid = max12 - min56 + 1

            for i in range(len(df)):
                tpc = df.iloc[i, tpc_ind]
                if tpc in [5, 6]:
                    df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + corrupmid
                else:
                    df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind]

        else:  # (sect[2] == 1): Track only in tpc56 and tpc910
            dfmid = df.loc[df['TPC'] == uni[0]]  # midstream tpc
            dfdown = df.loc[df['TPC'] == uni[1]]  # downstream tpc
            max56 = max(dfmid['Wirenum'])
            min910 = min(dfdown['Wirenum'])
            corrmiddown = max56 - min910 + 1

            for i in range(len(df)):
                tpc = df.iloc[i, tpc_ind]
                if tpc in [9, 10]:
                    df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + corrmiddown + 1148
                else:
                    df.iloc[i, planecorr_ind] = df.iloc[i, wirenum_ind] + 1148


def Plane_Corr(df, plane):
    """Combines plane corrections for all 3 planes into 1 function for easier readability"""
    if plane == 2:
        Plane_2_Correction(df)
    elif plane == 1:
        Plane_1_Corr(df)
    else:
        Plane_0_Corr(df)


def name_extract(filename):
    """Extract particle name from the input filename - used for names of saved graphs"""
    prev = 1
    particle = ""
    for i in filename:
        if (prev == 1) & (i != "_"):
            particle += i
        else:
            prev = 0
            exit()
    return particle


def BinData(df, incol, numbins, binmax, binmin):
    """Bin data (from incol) into numbins between binmin and binmax and return a new column which contains the halfway point of the bin that row is in."""
    binwidth = (binmax - binmin) / float(numbins)
    bins = []
    midpoints = []
    for i in range(0, numbins + 1):
        bins.append(binmin)
        midpoints.append(binmin + binwidth / 2.0)
        binmin += binwidth
    del midpoints[-1]

    return pd.cut(df[incol], bins, labels=midpoints, retbins=False)


def midind(df, i):
    planecorr_ind = df.columns.get_loc("PlaneCorrected")
    return int(df.iloc[i, planecorr_ind])


def binind(binpoint, numbins, binstart, binend):
    binwidth = (binend - binstart) / float(numbins)
    return int(((binpoint - binstart) / float(binwidth)) - 0.5)


def trackgraph(data, PlotLocation, size):
    xlen = size[0] / 100.0
    ylen = size[1] / 100.0
    colour = cm.jet
    # colour = 'gray'
    fig = plt.figure(frameon=False)  # Remove the white space around the image. Ensures image only consists of data
    fig.set_size_inches(xlen, ylen)  # Set the size of the image, in inches
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # Ensures image only consists of data
    ax.set_axis_off()  # Ensures image only consists of data
    fig.add_axes(ax)  # Add the object to the figure
    ax.imshow(data, origin='lower', aspect='equal', cmap=colour)  # Plot the numpy array, with the colour map
    fig.savefig(PlotLocation, dpi=100)  # Save to the given location, at 100dpi.
    plt.close()


def Empty_Track_Plot(loc, size):
    emptydata = np.zeros(size)
    trackgraph(emptydata, loc, size)


if __name__ == "__main__":
    file = "kao_1000.csv"  # file to read event data from
    planes = [0, 1, 2]

    evtnum_start = 1
    evtnum_end = 1000
    evtlen = 540  # how many wires past the first hit of track one to go in the event image for each plane

    xpixels = 540  # width of the images in pixels
    ypixels = 540  # height of the images in pixels

    saveloc = "test ML event plotter/"  # directory to save images in

    # setting up bins
    # if (evtlen % xpixels != 0):
    # print("xpixels %d must be a factor of evtlen %d!" % (xpixels,evtlen))
    # sys.exit()

    # if (4500 % ypixels != 0):
    # print("ypixels %d must be a factor of 4500!" % (ypixels))
    # sys.exit()

    planenumbins = xpixels
    timenumbins = ypixels  # time number of bins

    """setting up split between test and train data - depreciated"""
    # prob_train = 0.3
    # total = evtnum_end - evtnum_start + 1
    # rand.seed(7)
    # test = rand.sample(range(evtnum_start,evtnum_end,1),  int(prob_train*total))

    # inputting csv file and splitting wireID column into Cryostat number, TPC number, Plane, Wire ID
    df = pd.read_csv(file)
    print("file loaded")
    CTPW_split(df, 7)
    print("split done")
    particle = name_extract(file)

    count = 0
    # loop over events in the range specified
    for evtnum in range(evtnum_start, evtnum_end + 1):
        if evtnum % 10 == 0:
            print("Running for %s \t Event Number: %d" % (particle, evtnum))

        evtdf = df[df.evtNum == evtnum]
        if len(evtdf) == 0:
            print("No event for evtnum %d!" % evtnum)
            continue

        # loop over planes in the event
        for plane in planes:

            planedf = evtdf[evtdf.Plane == plane]

            if len(planedf) == 0:
                print("No plane data for evtnum %d plane %d!" % (evtnum, plane))
                continue

            # find all the tracks that have data in the plane
            tracks = planedf['TrackID'].unique()

            # if the longest track is not recorded in this plane produce a blank image and move on
            if 1 not in tracks:
                print("No track 1 recorded for evtnum: %d plane: %d" % (evtnum, plane))
                loc = str(saveloc) + str(particle) + "_" + str(evtnum) + "_" + str(plane)
                Empty_Track_Plot(loc, [xpixels, ypixels])
                count += 1
                continue

            # empty dataframe to be filled with all the tracks in the plane after corrections.
            corr_plane_df = pd.DataFrame()

            # store for track data before placing in the new dataframe
            evtlist = []

            # loop over tracks in the plane
            for track in tracks:
                # print("evtnum: %d, track: %d, plane: %d" % (evtnum, track, plane))
                trackdf = planedf[planedf.TrackID == track]

                red_track_df = pd.DataFrame({'evtNum': trackdf.loc[:, 'evtNum'].values,
                                             'TrackID': trackdf.loc[:, 'TrackID'].values,
                                             'PeakTime': trackdf.loc[:, 'PeakTime'].values,
                                             'Integral': trackdf.loc[:, 'Integral'].values,
                                             'PDG': trackdf.loc[:, 'matchedTrueTrackPDG'].values,
                                             'TPC': trackdf.loc[:, 'TPC'].values,
                                             'Plane': trackdf.loc[:, 'Plane'].values,
                                             'Wirenum': trackdf.loc[:, 'Wirenum'].values})

                # apply plane corrections on all the tracks - different for each plane
                Plane_Corr(red_track_df, plane)
                evtlist.extend(red_track_df.values.tolist())

                # if track is the longest - trackID 1 is always the longest track in events
                if track == 1:
                    # store the earliest plane position recorded - the start of the image.
                    start = np.floor(min(red_track_df['PlaneCorrected']))
                    # store the end plane position recorded - used for debugging.
                    # end = int(max(trackdf['PlaneCorrected']))

            # empty np array of the event with an element for every pixel of the final image to be filled with integral values
            # plotdata = np.zeros((timenumbins, evtlen))
            plotdata = np.zeros((ypixels, xpixels))

            # corrected track data for the plane
            corr_plane_df = pd.DataFrame(evtlist,
                                         columns=['Integral', 'PDG', 'PeakTime', 'Plane', 'TPC', 'TrackID', 'Wirenum',
                                                  'evtNum', 'PlaneCorrected'])

            # find the max and min times that PeakTime holds for data that will be plotted.
            img_df = corr_plane_df.loc[
                (corr_plane_df['PlaneCorrected'] < (start + evtlen)) & (corr_plane_df['PlaneCorrected'] >= start)]

            # bump the colour of the tracks in the image, so they look different from the background colour - debugging
            # bump = max(img_df['Integral'])*0.05
            bump = 0

            maxtime = max(img_df[
                              'PeakTime']) + 0.1  # bin data returns Nan if unbinned time is exactly equal to max or min time. Ensure it is always above the max.
            mintime = min(img_df[
                              'PeakTime']) - 0.1  # bin data returns Nan if unbinned time is exactly equal to max or min time. Ensure it is always below the min.
            # maxtime = 4500
            # mintime = 0
            maxplane = start + evtlen + 0.1
            minplane = start - 0.1

            # Bin PeakTime
            plot_df = pd.DataFrame({'evtNum': img_df.loc[:, 'evtNum'].values,
                                    'TrackID': img_df.loc[:, 'TrackID'].values,
                                    'PeakTime': BinData(img_df, 'PeakTime', timenumbins, maxtime, mintime),  # bins PeakTime data. Set maxtime = 4500, mintime = 0 to remove.
                                    'Integral': img_df.loc[:, 'Integral'].values,
                                    'PDG': img_df.loc[:, 'PDG'].values,
                                    'TPC': img_df.loc[:, 'TPC'].values,
                                    'Plane': img_df.loc[:, 'Plane'].values,
                                    'Wirenum': img_df.loc[:, 'Wirenum'].values,
                                    'PlaneCorrected': BinData(img_df, 'PlaneCorrected', planenumbins, maxplane,
                                                              minplane)})

            # find all the tracks in the range to be plotted
            tracks = plot_df['TrackID'].unique()

            # print("Running for %s \t Event Number: %d \t Plane: %d" % (particle, evtnum, plane))
            # loop over tracks
            for track in [1]:

                plot_track_df = plot_df[(plot_df.TrackID == track)]

                # fill plot array
                peaktime_ind = plot_track_df.columns.get_loc("PeakTime")
                planecorrected_ind = plot_track_df.columns.get_loc("PlaneCorrected")
                integral_ind = plot_track_df.columns.get_loc("Integral")

                for i in range(len(plot_track_df)):
                    y_ind = int(binind(plot_track_df.iloc[i, peaktime_ind], timenumbins, mintime, maxtime))
                    x_ind = int(binind(plot_track_df.iloc[i, planecorrected_ind], planenumbins, minplane, maxplane))

                    plotdata[y_ind][x_ind] += plot_track_df.iloc[i, integral_ind] + bump

            # plot and save image
            loc = str(saveloc) + str(particle) + "_" + str(evtnum) + "_" + str(plane)
            trackgraph(plotdata, loc, [xpixels, ypixels])
            count += 1

    print("completed for %s produced %d images" % (particle, count))
