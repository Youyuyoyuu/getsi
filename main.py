import os
import sys
import math
import fnmatch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.widgets import Button
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import HDBSCAN
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy import read, read_inventory
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

def autopick_time_window(qdata, tdata, faz):
    def rms(array):
        return np.sqrt(np.sum(array**2)/len(array))
    if faz:
        pdata, _ = rotate_data(-qdata, tdata, faz)
    sta_len = 1 #
    lta_len = 20 #
    trigger = 5 #
    detrigger = 2 #
    if faz == 0:
        data = qdata
    else:
        data = pdata
    cft = recursive_sta_lta(data, int(sta_len/delta), int(lta_len/delta))
    window = trigger_onset(cft, trigger, detrigger)
    if len(window) == 0:
        raise ValueError('No expected time window.')
    else:
        snr = np.zeros(len(window))
        for i in range(len(window)):
            if window[i][0] != lta_len/delta:
                a_noise = rms(data[int(window[i][0]-20/delta) : int(window[i][0]-10*delta)])
                a_signal = rms(data[window[i][0] : window[i][1]+1])
                snr[i] = 20 * np.log10(a_signal/a_noise)
            else:
                snr[i] = -100
        opt_w = snr.argmax()
        window_start2 = window[opt_w][0]
        window_start1 = int(window_start2 - args.autopick/delta)
        window_end1 = window[opt_w][1]
        window_end2 = int(window_end1 + args.autopick/delta)

        return window_start1, window_start2, window_end1, window_end2       

def calculate_si(pdata, odata):
    '''Calculate splitting intensity and cross-correlation with time shift.'''
    global derp_norm, pdata_norm, odata_norm
    max_p = np.max(np.abs(pdata))
    pdata_norm = pdata / max_p
    odata_norm = odata / max_p
    npoint = np.size(pdata_norm)
    derp_norm = np.zeros(npoint)
    for k in range(1, npoint-1):
        derp_norm[k] = (pdata_norm[k+1]-pdata_norm[k-1]) / (2*delta)
    pdata_norm = pdata_norm[1:-1]
    odata_norm = odata_norm[1:-1]
    derp_norm = derp_norm[1:-1]
    sq_rmsdr = np.dot(derp_norm, derp_norm.T)
    si = -2 * np.dot(odata_norm, derp_norm.T) / sq_rmsdr
    npoint = np.size(pdata_norm)
    err = np.sqrt((np.dot(odata_norm, odata_norm.T)-0.25*sq_rmsdr*si**2) / npoint)
    shift_limit = int(2 / delta)
    cc = correlate(derp_norm, odata_norm, shift_limit)
    ccshift, ccvalue = xcorr_max(cc)

    return si, err, ccvalue, ccshift

def cluster_analysis():
    global window_start, window_end, si, start_opt, end_opt, labels, unique_labels, maxlabel, sidata, paz_opt, si_opt, err_opt, ccvalue_opt, ccshift_opt, qdata_window, tdata_window
    interval = int(max((window_start2-window_start1)/50, (window_end2-window_end1)/50))
    # interval = 5
    window_start = np.arange(window_start1, window_start2+1, interval).astype(int)
    window_end = np.arange(window_end1, window_end2+1, interval).astype(int)
    si = np.zeros((window_start.size, window_end.size))
    err = np.zeros(window_start.size*window_end.size)
    sidata = np.zeros((window_start.size*window_end.size, 3))
    for i in range(window_start.size):
        for j in range(window_end.size):
            ldata_window, qdata_window, tdata_window = ldata[window_start[i]:window_end[j]+1], qdata[window_start[i]:window_end[j]+1], tdata[window_start[i]:window_end[j]+1]
            paz = flinn(ldata_window, qdata_window, tdata_window)
            pdata_window, odata_window = rotate_data(-qdata_window, tdata_window, paz)
            si[i][j], err[i*window_end.size+j:], _, _ = calculate_si(pdata_window, odata_window)
            sidata[i*window_end.size+j:] = window_end[j]*delta, window_start[i]*delta, si[i][j]
    data_scaled = standard(sidata)
    hdb = HDBSCAN()
    hdb.fit(data_scaled)
    cut_dist = interval * delta * 1.25 # The coefficient just depends on experience
    labels = hdb.dbscan_clustering(cut_distance=cut_dist, min_cluster_size=window_start.size)
    unique_labels, counts = np.unique(labels, return_counts=True)
    maxlabel = unique_labels[counts[1:].argmax() + 1]
    mask = labels == maxlabel
    errline = np.sort(err[mask])[int(mask.sum() * 0.8)]
    for i in range(err.size):
        if mask[i] and err[i]>errline:
            labels[i] = -2
    unique_labels = np.append(unique_labels, -2)
    labels_2d = labels.reshape([window_start.size, window_end.size])
    indexlist = np.where(labels_2d == maxlabel)
    start_opt = int(round(np.mean(indexlist[0]))*interval + window_start1)
    end_opt = int(round(np.mean(indexlist[1]))*interval + window_end1)
    ldata_window, qdata_window, tdata_window = ldata[start_opt:end_opt+1], qdata[start_opt:end_opt+1], tdata[start_opt:end_opt+1]
    paz_opt = flinn(ldata_window, qdata_window, tdata_window)
    pdata_window, odata_window = rotate_data(-qdata_window, tdata_window, paz)
    si_opt, err_opt, ccvalue_opt, ccshift_opt = calculate_si(pdata_window, odata_window)

def flinn(ldata, qdata, tdata):
    '''Changed from obspy's source code.'''
    x = np.zeros((3, ldata.size), dtype=np.float64)
    # East for x0, here changed to T
    x[0, :] = tdata #
    # North for x1, here changed to R or -Q
    x[1, :] = -qdata
    # Z for x2, here changed to L
    x[2, :] = ldata #

    covmat = np.cov(x)
    eigvec, eigenval, v = np.linalg.svd(covmat)
    azimuth = math.degrees(math.atan2(eigvec[0][0], eigvec[1][0]))
    eve = np.sqrt(eigvec[0][0] ** 2 + eigvec[1][0] ** 2)
    incidence = math.degrees(math.atan2(eve, eigvec[2][0]))
    if azimuth < 0.0:
        azimuth = 360.0 + azimuth
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if azimuth > 180.0:
            azimuth -= 180.0
        else:
            azimuth += 180.0
    if azimuth > 180.0:
        azimuth -= 180.0

    return azimuth

def focal_polarization(M, theta, phi, r=1):
    '''Calculating theoretical polarization direction according to focal mechanism.'''
    # vp = 5.
    vs = 3.5
    rho = 3.
    theta = theta * np.pi / 180
    phi = phi * np.pi / 180
    [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp] = M
    M = np.matrix([[Mtt, -Mtp, Mrt],
                   [-Mtp, Mpp, -Mrp],
                   [Mrt, -Mrp, Mrr]])
    gama = np.array([np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)]).reshape(1,3)
    p_s = np.array([np.cos(theta)*np.cos(phi),
                    np.cos(theta)*np.sin(phi),
                    -np.sin(theta)]).reshape(1,3)
    phi_s = np.array([-np.sin(phi),
                      np.cos(phi),
                      0]).reshape(1,3)
    # u_p = gama * M * gama.T / (4*np.pi*vp**3*rho*r)
    u_sv = p_s * M * gama.T / (4*np.pi*vs**3*rho*r)
    u_sh = phi_s * M * gama.T / (4*np.pi*vs**3*rho*r)
    paz = np.arctan2(u_sh, u_sv) * 180 / np.pi
    paz = 180 - paz

    return paz[0, 0]

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filespath', help='absolute or relative path of data files')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--autopick', help='automatically pick time window (a timespan is needed)', type=float)
    group.add_argument('-m', '--manual', help='choose to pick 2 or 4 timepoints', choices=[2, 4], type=int)
    parser.add_argument('-f', '--filter', help='default bandpass filter (0.02-0.2Hz)', action='store_true')
    parser.add_argument('-i', '--inventory', help='path of the inventory with metadata of channels')
    parser.add_argument('-p', '--phase', help='expected phase (default: S)', default='S')
    parser.add_argument('-r', '--preprocess', help='detrend, taper and remove instrument response', action='store_true')
    # parser.add_argument('--force', help='ingnore previous execution and force to run the code once', action='store_true')
    args = parser.parse_args()

    return args

def moment_tensor():
    '''Search the moment tensor of a certain event.
    If not exist, return None.
    '''
    evt_search = event_time[ : 16]
    evt_search = evt_search.replace('-', '/')
    evt_search = evt_search.replace('T', ' ')
    fm = open('./jan76_dec20.ndk', 'r')
    text = np.array(fm.readlines())
    mo = fnmatch.filter(text, '*'+evt_search+'*')
    if len(mo) < 1:
        M = None
    else:
        index = np.where(text==mo[0])[0][0]
        # event_info = text[index].split()
        # date, time, lat, lon, dep, mag = event_info[1:7]
        moment = text[index+3].split()
        expo, M= moment[0], moment[1 : : 2]
        M = [float(m) for m in M]

    return M

def pick_time_window2(qdata, tdata, faz):
    if faz != 0:
        pdata, odata = rotate_data(-qdata, tdata, faz)
    timepoint = np.zeros(2)
    def click(event):
        if timepoint[0] == 0:
            timepoint[0] = event.xdata
            ax1.axvline(x=event.xdata, color='teal', lw=0.3)
        elif timepoint[1] == 0:
            timepoint[1] = event.xdata
            ax1.axvline(x=event.xdata, color='darkslateblue', lw=0.3)
        else:
            pass
        fig.canvas.draw()
    def repick(event):
        ax1.axvline(x=timepoint[0], color='white', lw=0.5)
        ax1.axvline(x=timepoint[1], color='white', lw=0.5)
        fig.canvas.draw()
        timepoint[0], timepoint[1] = 0, 0
    def next(event):
        plt.close()
    def quit(event):
        sys.exit(0)

    npts = qdata.size
    fig = plt.figure(figsize=[16, 6])
    ax0 = fig.add_axes([0.37, 0.025, 0.05, 0.05], frame_on=False, xticks=[], yticks=[])
    ax0.text(0.5, 0.5, pair_name)
    ax1 = fig.add_axes([0.03, 0.15, 0.95, 0.80])
    x = np.arange(npts)
    if faz == 0:
        ax1.plot(x, qdata, color='firebrick', label='Q')
        ax1.plot(x, tdata, color='midnightblue', linestyle='dashed', label='T')
    else:
        ax1.plot(x, pdata, color='crimson', label='P')
        ax1.plot(x, odata, color='dimgrey', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    # ax1.axvline(x=30/delta, color='darkgrey', linestyle='dotted') ######
    ax1.set_xlim(0, npts)
    ax1.legend()
    xlabel_original = np.arange(0, npts, 10/delta)
    xlabel_changed = xlabel_original*delta
    ax1.set_xticks(xlabel_original, xlabel_changed)
    for i in range(2):
        fig.canvas.mpl_connect('button_press_event', click)
    ax2 = fig.add_axes([0.87, 0.035, 0.05, 0.05])
    btn1 = Button(ax2, 'repick', color='darkgrey')
    ax3 = fig.add_axes([0.93, 0.035, 0.05, 0.05])
    btn2 = Button(ax3, 'next', color='darkgrey')
    ax4 = fig.add_axes([0.81, 0.035, 0.05, 0.05])
    btn3 = Button(ax4, 'quit', color='darkgrey')
    btn1.on_clicked(repick)
    btn2.on_clicked(next)
    btn3.on_clicked(quit)
    plt.show()
    window_start, window_end = int(timepoint[0]), int(timepoint[1])
    return window_start, window_end

def pick_time_window4(qdata, tdata, faz):
    '''The input stream must be in LQT order.
    If theoretical polarization direction is not provided,
    only Q and T's waveforms will be displayed.
    '''
    if faz:
        pdata, odata = rotate_data(-qdata, tdata, faz)
    timepoint = np.zeros(4)
    def click(event):
        if timepoint[0] == 0:
            timepoint[0] = event.xdata
            ax1.axvline(x=event.xdata, color='teal', lw=0.3)
        elif timepoint[1] == 0:
            timepoint[1] = event.xdata
            ax1.axvline(x=event.xdata, color='teal', lw=0.3)
        elif timepoint[2] == 0:
            timepoint[2] = event.xdata
            ax1.axvline(x=event.xdata, color='darkslateblue', lw=0.3)
        elif timepoint[3] == 0:
            timepoint[3] = event.xdata
            ax1.axvline(x=event.xdata, color='darkslateblue', lw=0.3)
        else:
            pass
        fig.canvas.draw()
    def repick(event):
        ax1.axvline(x=timepoint[0], color='white', lw=0.5)
        ax1.axvline(x=timepoint[1], color='white', lw=0.5)
        ax1.axvline(x=timepoint[2], color='white', lw=0.5)
        ax1.axvline(x=timepoint[3], color='white', lw=0.5)
        fig.canvas.draw()
        timepoint[0], timepoint[1], timepoint[2], timepoint[3] = 0, 0, 0, 0
    def next(event):
        plt.close()
    def quit(event):
        sys.exit(0)

    npts = qdata.size
    fig = plt.figure(figsize=[16, 6])
    ax0 = fig.add_axes([0.37, 0.025, 0.05, 0.05], frame_on=False, xticks=[], yticks=[])
    ax0.text(0.5, 0.5, pair_name)
    ax1 = fig.add_axes([0.03, 0.15, 0.95, 0.80])
    x = np.arange(npts)
    if faz is None:
        ax1.plot(x, qdata, color='firebrick', label='Q')
        ax1.plot(x, tdata, color='midnightblue', linestyle='dashed', label='T')
    else:
        ax1.plot(x, pdata, color='crimson', label='P')
        ax1.plot(x, odata, color='dimgrey', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    # ax1.axvline(x=30/delta, color='darkgrey', linestyle='dotted') ######
    ax1.set_xlim(0, npts)
    ax1.legend()
    xlabel_original = np.arange(0, npts, 10/delta)
    xlabel_changed = xlabel_original*delta
    ax1.set_xticks(xlabel_original, xlabel_changed)
    for i in range(4):
        fig.canvas.mpl_connect('button_press_event', click)
    ax2 = fig.add_axes([0.87, 0.035, 0.05, 0.05])
    btn1 = Button(ax2, 'repick', color='darkgrey')
    ax3 = fig.add_axes([0.93, 0.035, 0.05, 0.05])
    btn2 = Button(ax3, 'next', color='darkgrey')
    ax4 = fig.add_axes([0.81, 0.035, 0.05, 0.05])
    btn3 = Button(ax4, 'quit', color='darkgrey')
    btn1.on_clicked(repick)
    btn2.on_clicked(next)
    btn3.on_clicked(quit)
    plt.show()
    window_start1, window_start2, window_end1, window_end2 = int(timepoint[0]), int(timepoint[1]), int(timepoint[2]), int(timepoint[3])

    return window_start1, window_start2, window_end1, window_end2

def prepare():
    '''Rotate from original ZNE to LQT components, 
    meanwhile write incident and take-off angle into new files.
    '''
    global tak, az
    model = TauPyModel('iasp91')
    hdr = st[0].stats.sac
    if hdr.evdp > 700: # depth in m
        arrivals = model.get_ray_paths(hdr.evdp/1000, hdr.gcarc, [args.phase])
    elif hdr.evdp <= 700: # depth in km
        arrivals = model.get_ray_paths(hdr.evdp, hdr.gcarc, [args.phase])
    else:
        raise ValueError('Header lost depth.')
    inc = arrivals[0].incident_angle
    tak = arrivals[0].takeoff_angle
    baz = hdr.baz
    az = hdr.az
    if args.filter:
        st.filter('bandpass', freqmin=0.02, freqmax=0.2, corners=2, zerophase=True)
    try:
        try:
            zdata = st.select(component = 'Z')[0].data
        except:
            zdata = st.select(component = 'U')[0].data
        ndata = st.select(component = 'N')[0].data
        edata = st.select(component = 'E')[0].data
    except:
        raise ValueError('Must be ZNE channels.')
    ldata, qdata, tdata = zne2lqt(zdata, ndata, edata, baz, inc)

    return ldata, qdata, tdata

def preprocess():
    '''Preprocess the raw data.
    If there is no response for a certain instrument 
    either in a collection or in a single file, 
    search and write down one for this instrument individually.
    '''
    os.makedirs('./cache/resp/', exist_ok=True)
    st.detrend('constant')
    st.detrend('linear')
    st.taper(max_percentage = 0.05)
    try:
        if args.inventory:
            inv = read_inventory('../inv.xml')
        else:
            inv = read_inventory('./cache/resp/' + station_code + '.xml')
    except:
        client = Client('IRIS')
        channel = st[0].stats.channel[ : 2]+'*'
        starttime = st[0].stats.starttime
        endtime = st[0].stats.endtime
        if location is None:
            location_search = '--'
        else:
            location_search = location
        client.get_stations(starttime=starttime, endtime=endtime,
                            network=net, station=station, location=location_search, channel=channel,
                            level='response', filename='./cache/resp/'+station_code+'.xml', format='xml')
        inv = read_inventory('./cache/resp/' + station_code + '.xml')
    st.remove_response(inventory = inv)

def process():
    global window_start1, window_start2, window_end1, window_end2, faz
    if M is None:
        faz = None
    else:
        faz = focal_polarization(M, tak, az)
    if args.autopick:
        window_start1, window_start2, window_end1, window_end2 = autopick_time_window(qdata, tdata, faz)
    elif args.manual == 2:
        window_start1, window_end2 = pick_time_window2(qdata, tdata, faz)
        window_start2, window_end1 = int((window_start1+window_end2)/2)-10, int((window_start1+window_end2)/2)+10
    else:
        window_start1, window_start2, window_end1, window_end2 = pick_time_window4(qdata, tdata, faz)
    cluster_analysis()
    result_plot()

def result_plot():
    '''Show up the final results in a figure.'''
    def back(event):
        plt.close()
        process()
    def quit(event):
        sys.exit(0)

    fig = plt.figure(figsize=[16, 10])

    ax0 = fig.add_axes([0.05, 0.54, 0.425, 0.38], frame_on=False, xticks=[], yticks=[]) # information
    text_content = 'Event: '+event_time+'\n'\
                    +'Station: '+station_code+'\n'\
                    +'SI = '+str(round(si_opt, 2))+'\u00B1'+str(round(err_opt, 2))
    ax0.text(0.3, 1.0, text_content, bbox=dict(facecolor='white', linewidth=0.5),ha='left', va='top',fontsize=12, linespacing=1.8)
    ax1 = fig.add_axes([0.05, 0.54, 0.425, 0.28]) # waveform
    pdata, odata = rotate_data(-qdata, tdata, paz_opt)
    x = np.arange(pdata.size)
    ax1.plot(x, pdata, color='crimson', label='P')
    ax1.plot(x, odata, color='dimgrey', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax1.axvline(x=30/delta, color='darkgrey', linestyle='dotted')
    max_amp = np.max(np.abs(pdata))
    rect1_1 = pch.Rectangle((window_start1, -2*max_amp), window_start2-window_start1, 4*max_amp, color='teal', alpha=0.05)
    ax1.add_patch(rect1_1)
    rect2 = pch.Rectangle((window_end1, -2*max_amp), window_end2-window_end1, 4*max_amp, color='darkslateblue', alpha=0.05)
    ax1.add_patch(rect2)
    ax1.axvline(x=start_opt, color='teal')
    ax1.axvline(x=end_opt, color='darkslateblue')
    ax1.set_xlim(0, pdata.size)
    ax1.legend()
    ax1.xaxis.set_major_locator(MultipleLocator(20/delta))
    ax1.xaxis.set_minor_locator(MultipleLocator(5/delta))
    xlabel_original = np.arange(0, pdata.size, 20/delta)
    xlabel_changed = xlabel_original*delta
    ax1.set_xticks(xlabel_original, xlabel_changed)

    ax2_1 = fig.add_axes([0.515, 0.54, 0.2, 0.32], projection='3d')
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.8, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 0.2]
        elif k == -2:
            col = [0.8, 0.1, 0.1, 0.1]
        elif k == maxlabel:
            col = [0.8, 0.1, 0.1, 0.9]
        data_p = sidata[labels==k]
        ax2_1.scatter(data_p[:, 0], data_p[:, 1], data_p[:, 2], color=tuple(col))
    ax2_1.set_title('Clusters: %d' % (len(unique_labels)-2))
    ax2_2 = fig.add_axes([0.75, 0.54, 0.2, 0.32]) # distribution of SI
    si_distribution = ax2_2.contour(window_end*delta, window_start*delta, si, colors='k')
    ax2_2.xaxis.set_major_locator(MultipleLocator(2))
    ax2_2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2_2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2_2.yaxis.set_ticks_position('right')
    ax2_2.axhline(start_opt*delta, color='black')
    ax2_2.axvline(end_opt*delta, color='black')
    ax2_2.clabel(si_distribution, inline=True)
    ax2_2.set_title('SI')

    ax3 = fig.add_axes([0.05, 0.08, 0.425, 0.38]) # particle motion
    mid = (np.average(tdata_window), np.average(-qdata_window))
    m = max([max(np.absolute(-qdata_window)), max(np.absolute(tdata_window))])
    ax3.plot(tdata_window,-qdata_window, color='grey')
    ax3.scatter(tdata_window[0], -qdata_window[0], color='grey')
    ax3.axline(mid, slope=np.tan((90-paz_opt)*np.pi/180), color='crimson', label='calc_pd')
    if faz:
        ax3.axline(mid, slope=np.tan((90-faz)*np.pi/180), color='mediumpurple', label='theo_pd')
        ax3.set_title('Polarization direction: %.2f\u00b0 (%.2f\u00b0)' % (paz_opt, faz))
    else:
        ax3.set_title('Polarization direction: %.2f\u00b0' % paz_opt)
    ax3.set_xlim(-1.2*m, 1.2*m)
    ax3.set_ylim(-1.2*m, 1.2*m)
    ax3.set_xlabel('T')
    ax3.set_ylabel('-Q')
    ax3.legend()

    ax4 = fig.add_axes([0.525, 0.08, 0.425, 0.38]) # time window
    ax4.plot(np.arange(pdata_norm.size), pdata_norm, color='crimson', label='P', linestyle=(0, (1, 5)))
    ax4.plot(np.arange(odata_norm.size)+ccshift_opt, odata_norm, color='black', ls='dashed', label='O')
    ax4.plot(np.arange(derp_norm.size), -derp_norm*si_opt/2, color='brown', label='- 0.5P\' \u00B7 SI')
    ax4.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax4.axvline(x=0, color='darkgrey', linestyle='dotted')
    ax4.axvline(x=pdata_norm.size, color='darkgrey', linestyle='dotted')
    ax4.xaxis.set_major_locator(MultipleLocator(1/delta))
    ax4.set_xticklabels([])
    ax4.text(0.01*pdata_norm.size, -0.98, str(round(start_opt*delta, 2)),ha='left', va='bottom', color='darkgrey')
    ax4.text(0.99*pdata_norm.size, -0.98, str(round(end_opt*delta, 2)),ha='right', va='bottom', color='darkgrey')
    ax4.set_title('Cross-correlation: %.2f (shift=%.2fs)' % (ccvalue_opt, ccshift_opt*delta))
    ax4.set_ylim(-1, 1)
    ax4.legend()

    ax5 = fig.add_axes([0.9, 0.015, 0.05, 0.04])
    btn1 = Button(ax5, 'back', color='darkgrey')
    btn1.on_clicked(back)
    ax6 = fig.add_axes([0.84, 0.015, 0.05, 0.04])
    btn2 = Button(ax6, 'quit', color='darkgrey')
    btn2.on_clicked(quit)
    plt.show()

def rotate_data(y, x, theta):
    '''Rotate theta(degrees)'''
    theta = math.radians(theta)
    y_prime = y*np.cos(theta) + x*np.sin(theta)
    x_prime = -y*np.sin(theta) + x*np.cos(theta)

    return y_prime, x_prime

def standard(data):
    timespan = max(window_start[-1]-window_start[0], window_end[-1]-window_end[0]) * delta
    sirange = np.max(sidata[:, 2]) - np.min(sidata[:, 2])
    scale = timespan / sirange
    data_scaled = data.copy()
    data_scaled[:, 2] = data_scaled[:, 2] * scale
    return data_scaled

def zne2lqt(z, n, e, baz, inc):
    '''rotate ZNE to LQT,
    inc = 0 means vertically incident wave.
    '''
    inc = math.radians(inc)
    baz = math.radians(baz)
    l = np.cos(inc)*z - np.sin(inc)*np.sin(baz)*e - np.sin(inc)*np.cos(baz)*n
    q = np.sin(inc)*z + np.cos(inc)*np.sin(baz)*e + np.cos(inc)*np.cos(baz)*n
    t = -np.cos(baz)*e + np.sin(baz)*n

    return l, q, t

if __name__ == '__main__':
    args = get_arguments()
    st = read(args.filespath+'*')
    # st.decimate(2, no_filter=True)
    # st.interpolate(100)
    net = st[0].stats.network
    station = st[0].stats.station
    location = st[0].stats.location
    station_code = net+'.'+station+'.'+location
    event_time = (st[0].stats.starttime+st[0].stats.sac.o).format_iris_web_service()
    pair_name = event_time+'-'+station_code
    if args.preprocess:
        preprocess()
    ldata, qdata, tdata = prepare()
    M = moment_tensor()
    delta = st[0].stats.delta
    process()