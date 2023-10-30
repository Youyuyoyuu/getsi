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
from sklearn.cluster import DBSCAN
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy import read, read_inventory
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.filter import envelope, lowpass
from distaz import DistAz

def add_header():
    global net, station, location, delta, station_code, event_time, pair_name, hdr, inc, tak
    net = st[0].stats.network
    station = st[0].stats.station
    location = st[0].stats.location
    delta = st[0].stats.delta
    station_code = net+'.'+station+'.'+location
    hdr = st[0].stats.sac
    try:
        event_time = st[0].stats.starttime+hdr.o-hdr.b
        date, time = np.loadtxt('../cat.lst', usecols=(0, 1), dtype={'names': ('date', 'time'), 'formats': ('U10', 'U12')}, unpack=True)
        for i in range(date.size):
            if abs(event_time-UTCDateTime(date[i]+' '+time[i])) <= 5: break
        event_time = str(UTCDateTime(date[i]+' '+time[i]))
    except:
        raise ValueError('Header lacks the origin time of the event.')
    pair_name = event_time+'-'+station_code
    try:
        stla, stlo, evla, evlo, evdp = hdr.stla, hdr.stlo, hdr.evla, hdr.evlo, hdr.evdp
    except:
        raise ValueError('Header lacks geographic infomation of the station or the event.')
    results = DistAz(stla, stlo, evla, evlo)
    hdr.gcarc, hdr.az, hdr.baz = results.getDelta(), results.getAz(), results.getBaz()
    model = TauPyModel('iasp91')
    try:
        if hdr.evdp > 700: # depth in m
            arrivals = model.get_ray_paths(hdr.evdp/1000, hdr.gcarc, [args.phase])
        else: # depth in km
            arrivals = model.get_ray_paths(hdr.evdp, hdr.gcarc, [args.phase])
    except:
        raise ValueError('Header lacks the depth of the event.')
    inc, tak, hdr.a = arrivals[0].incident_angle, arrivals[0].takeoff_angle, hdr.o+arrivals[0].time

def autopick_time_window(qdata, tdata, paz):
    def rms(array):
        return np.sqrt(np.sum(array**2)/len(array))
    pdata, _ = rotate_data(-qdata, tdata, paz)
    p_envelope = envelope(pdata)
    noise = rms(p_envelope[int(10/delta) : int(30/delta)])
    pick = np.where(p_envelope>3*noise)[0]
    start = pick[0]
    for i in np.arange(1, pick.size-1):
        if pick[i]-pick[i-1]>1 and pick[i+1]-pick[i]==1:
            start = pick[i]
        if pick[i]-pick[i-1]==1 and pick[i+1]-pick[i]>1:
            if pick[i] - start >= 10/delta:
                break
            else:
                continue
    window_len = pick[i] - start
    if window_len < 10/delta:
        raise ValueError
    window_start1 = int(start-window_len/3)
    window_start2 = int(start+window_len/3)
    window_end1 = int(pick[i]-window_len/3)
    window_end2 = int(pick[i]+window_len/3)

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
    shift_limit = int(0 / delta)
    cc = correlate(derp_norm, odata_norm, shift_limit)
    ccshift, ccvalue = xcorr_max(cc)

    return si, err, abs(ccvalue), ccshift

def cluster_analysis():
    global window_start, window_end, si, err, start_opt, end_opt, labels, unique_labels, maxlabel, proportion, sidata, paz_opt, si_opt, err_opt, ccvalue_opt, ccshift_opt, qdata_window, tdata_window
    window_start = np.linspace(window_start1, window_start2+1, 21).astype(int)
    window_end = np.linspace(window_end1, window_end2+1, 21).astype(int)
    si = np.zeros(window_start.size*window_end.size)
    err = np.zeros(window_start.size*window_end.size)
    sidata = np.zeros((window_start.size*window_end.size, 3))
    for i in range(window_start.size):
        for j in range(window_end.size):
            ldata_window, qdata_window, tdata_window = ldata[window_start[i]:window_end[j]+1], qdata[window_start[i]:window_end[j]+1], tdata[window_start[i]:window_end[j]+1]
            paz = flinn(ldata_window, qdata_window, tdata_window)
            pdata_window, odata_window = rotate_data(-qdata_window, tdata_window, paz)
            si[i*window_end.size+j], err[i*window_end.size+j], _, _ = calculate_si(pdata_window, odata_window)
            sidata[i*window_end.size+j:] = window_end[j]*delta, window_start[i]*delta, si[i*window_end.size+j]
    data_scaled = standard(sidata)
    hdb = DBSCAN(eps=0.075, min_samples=5).fit(data_scaled)
    labels = hdb.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    # print('station: '+station_code+'------------')
    # for i in unique_labels:
    #     if unique_labels[i] == -1:
    #         continue
    #     mask = labels == unique_labels[i]
    #     sigma_c = np.var(si[mask])
    #     sigma_d = 1/np.sum((1/err[mask])**2)
    #     err_mean = np.mean(err[mask])
    #     print('label: %d\ncounts=%d, err=%.2f\nsigma_c=%.4f, sigma_d=%.4f\n' % (unique_labels[i], counts[i], err_mean, sigma_c, sigma_d))

    maxcounts = 1
    for i in unique_labels:
        if unique_labels[i]==-1 or counts[i]<40:
            continue
        if counts[i] >= 1.8*maxcounts:
            maxlabel = unique_labels[i]
            maxcounts = counts[i]
            mask = labels == unique_labels[i]
            minstd = np.std(si[mask])
        elif counts[i] > maxcounts:
            mask = labels == unique_labels[i]
            std = np.std(si[mask])
            if std <= minstd:
                maxlabel = unique_labels[i]
                maxcounts = counts[i]
                minstd = std
        elif counts[i] > 0.8*maxcounts:
            mask = labels == unique_labels[i]
            std = np.std(si[mask])
            if std < 0.8*minstd:
                maxlabel = unique_labels[i]
                maxcounts = counts[i]
                minstd = std
    mask = labels == maxlabel
    proportion = np.sum(mask) / (window_start.size*window_end.size)
    mean, std = np.mean(si[mask]), np.std(si[mask])
    for i in range(window_start.size*window_end.size):
        if mask[i] and abs(si[i]-mean)>std:
            labels[i] = -2
    unique_labels = np.append(unique_labels, -2)
    labels_2d = labels.reshape([window_start.size, window_end.size])
    indexlist = np.where(labels_2d == maxlabel)
    mean = np.mean(indexlist[0])
    ceil = math.ceil(mean)
    floor = math.floor(mean)
    if ceil == floor:
        start_opt = window_start[ceil]
    else:
        start_opt = int(window_start[ceil]*(mean-floor) + window_start[floor]*(ceil-mean))
    mean = np.mean(indexlist[1])
    ceil = math.ceil(mean)
    floor = math.floor(mean)
    if ceil == floor:
        end_opt = window_end[ceil]
    else:
        end_opt = int(window_end[ceil]*(mean-floor) + window_end[floor]*(ceil-mean))
    ldata_window, qdata_window, tdata_window = ldata[start_opt:end_opt+1], qdata[start_opt:end_opt+1], tdata[start_opt:end_opt+1]
    paz_opt = flinn(ldata_window, qdata_window, tdata_window)
    pdata_window, odata_window = rotate_data(-qdata_window, tdata_window, paz_opt)
    si_opt, err_opt, ccvalue_opt, ccshift_opt = calculate_si(pdata_window, odata_window)

def flinn(ldata, qdata, tdata):
    '''Changed from obspy's source code.'''
    ldata = lowpass(ldata, freq=0.05, df=1/delta, corners=2, zerophase=True)
    qdata = lowpass(qdata, freq=0.05, df=1/delta, corners=2, zerophase=True)
    tdata = lowpass(tdata, freq=0.05, df=1/delta, corners=2, zerophase=True)
    x = np.zeros((3, ldata.size), dtype=np.float64)
    # East for x0, here changed to T
    x[0, :] = tdata #
    # North for x1, here changed to R or -Q
    x[1, :] = -qdata
    # Z for x2, here changed to L
    x[2, :] = ldata #

    covmat = np.cov(x)
    eigvec, _, _ = np.linalg.svd(covmat)
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
    parser.add_argument('-a', '--autopick', help='automatically pick time window', action='store_true')
    parser.add_argument('-f', '--filter', help='default bandpass filter', action='store_true')
    parser.add_argument('-i', '--inventory', help='path of the inventory with metadata of channels')
    parser.add_argument('-p', '--phase', help='expected phase (default: S)', default='S')
    parser.add_argument('-r', '--preprocess', help='detrend, taper and remove instrument response', action='store_true')
    parser.add_argument('-t', '--plot3d', help='show 3D figure instead of 2D', action='store_true')
    args = parser.parse_args()

    return args

def moment_tensor():
    '''Search the moment tensor of a certain event.
    If not exist, return None.
    '''
    evt_search = event_time[ : 10]
    evt_search = evt_search.replace('-', '/')
    # evt_search = evt_search.replace('T', ' ')
    fm = open('./jan76_dec20.ndk', 'r')
    text = np.array(fm.readlines())
    mo = fnmatch.filter(text, '*'+evt_search+'*')
    if len(mo) < 1:
        M = None
    else:
        for i in range(len(mo)):
            index = np.where(text==mo[i])[0][0]
            event_info = text[index].split()
            # date, time, lat, lon, dep, mag = event_info[1:7]
            date, time, lat, lon = event_info[1 : 5]
            time_diff = abs(UTCDateTime(event_time)-UTCDateTime(date+' '+time))
            epic_dist = DistAz(float(lat), float(lon), hdr.evla, hdr.evlo).getDelta()
            if time_diff>24*60*60 or epic_dist>2:
                continue
            else:
                break
        moment = text[index+3].split()
        expo, M= moment[0], moment[1 : : 2]
        M = [float(m) for m in M]

    return M

def pick_time_window(qdata, tdata, faz):
    '''The input stream must be in LQT order.
    If theoretical polarization direction is not provided,
    only Q and T's waveforms will be displayed.
    '''
    if faz is not None:
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
    ax1.axvline(x=(hdr.a-hdr.b)/delta, color='darkgrey', linestyle='dotted')
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
    if args.filter:
        st.filter('bandpass', freqmin=0.02, freqmax=0.125, corners=2, zerophase=True)
    try:
        zdata = st.select(component = 'Z')[0].data
        ndata = st.select(component = 'N')[0].data
        edata = st.select(component = 'E')[0].data
    except:
        raise ValueError('Must be ZNE channels.')
    ldata, qdata, tdata = zne2lqt(zdata, ndata, edata, hdr.baz, inc)

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
            inv = read_inventory(args.inventory)
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
    global window_start1, window_start2, window_end1, window_end2, faz, paz_ini
    if M is None:
        faz = None
    else:
        faz = focal_polarization(M, tak, hdr.az)
    start_ini = int((hdr.a-hdr.b)/delta)
    end_ini = int(start_ini+30/delta)
    paz_ini = flinn(ldata[start_ini : end_ini], qdata[start_ini : end_ini], tdata[start_ini : end_ini])
    if args.autopick is True:
        window_start1, window_start2, window_end1, window_end2 = autopick_time_window(qdata, tdata, paz_ini)
    else:
        window_start1, window_start2, window_end1, window_end2 = pick_time_window(qdata, tdata, paz_ini)
    cluster_analysis()
    result_plot()

def result_plot():
    '''Show up the final results in a figure.'''
    def abandon(event):
        plt.close()
    def wait(event):
        wait_list.append(station_code)
        plt.close()
    def quit(event):
        sys.exit(0)
    def record(event):
        if os.path.exists('./results.lst') is False:
            f = open('./results.lst', 'w')
            f.write('# event station SI error cc proportion\n'
                    +event_time+' '+station_code+' '+str(round(si_opt, 2))+' '+str(round(err_opt, 2))+' '+str(round(ccvalue_opt, 2))+' '+str(round(proportion, 2))+'\n')
            f.close()
        else:
            f = open('./results.lst', 'a+')
            f.write(event_time+' '+station_code+' '+str(round(si_opt, 2))+' '+str(round(err_opt, 2))+' '+str(round(ccvalue_opt, 2))+' '+str(round(proportion, 2))+'\n')
            f.close()
        plt.close()

    fig = plt.figure(figsize=[16, 10])

    ax0 = fig.add_axes([0.05, 0.54, 0.425, 0.38], frame_on=False, xticks=[], yticks=[]) # information
    text_content = 'Event: '+event_time+'\n'\
                    +'Station: '+station_code+'\n'\
                    +'SI = '+str(round(si_opt, 2))+'\u00B1'+str(round(err_opt, 2))
    if station == 'DJT':
        ax0.text(0.3, 1.0, text_content, bbox=dict(facecolor='grey', linewidth=0.5),ha='left', va='top',fontsize=12, linespacing=1.8)
    else:
        ax0.text(0.3, 1.0, text_content, bbox=dict(facecolor='white', linewidth=0.5),ha='left', va='top',fontsize=12, linespacing=1.8)
    ax1 = fig.add_axes([0.05, 0.54, 0.425, 0.28]) # waveform
    pdata, odata = rotate_data(-qdata, tdata, paz_opt)
    x = np.arange(pdata.size)
    ax1.plot(x, pdata, color='crimson', label='P')
    ax1.plot(x, odata, color='dimgrey', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax1.axvline(x=(hdr.a-hdr.b)/delta, color='darkgrey', linestyle='dotted')
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

    ax2_0 = fig.add_axes([0.63, 0.8, 0.22, 0.2], frame_on=False, xticks=[], yticks=[])
    cluster_number = len(unique_labels)
    if -1 in unique_labels:
        cluster_number -= 1
    if -2 in unique_labels:
        cluster_number -= 1
    ax2_0.text(0.36, 0.5, 'Clusters: '+str(cluster_number), fontsize=14)
    ax2_1 = fig.add_axes([0.53, 0.54, 0.2, 0.32]) # distribution of SI
    si_distribution = ax2_1.contour(window_end*delta, window_start*delta, si.reshape([window_start.size, window_end.size]), colors='k')
    colors = [plt.cm.Spectral(each) for each in np.linspace(0.6, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 0.1]
        elif k == -2:
            col = [0.8, 0.1, 0.1, 0.2]
        elif k == maxlabel:
            col = [0.8, 0.1, 0.1, 0.5]
        else:
            col = list(col)
            col[3] = 0.3
        data_p = sidata[labels==k]
        ax2_1.scatter(data_p[:, 0], data_p[:, 1], color=tuple(col))
    ax2_1.xaxis.set_minor_locator(MultipleLocator(1))
    ax2_1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2_1.yaxis.set_ticks_position('left')
    ax2_1.axhline(start_opt*delta, color='crimson')
    ax2_1.axvline(end_opt*delta, color='crimson')
    ax2_1.clabel(si_distribution, inline=True)
    ax2_1.set_title('SI')
    if args.plot3d is True:
        ax2_2 = fig.add_axes([0.75, 0.54, 0.2, 0.32], projection='3d')
        colors = [plt.cm.Spectral(each) for each in np.linspace(0.6, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 0.2]
            elif k == -2:
                col = [0.8, 0.1, 0.1, 0.2]
            elif k == maxlabel:
                col = [0.8, 0.1, 0.1, 0.9]
            data_p = sidata[labels==k]
            ax2_2.scatter(data_p[:, 0], data_p[:, 1], data_p[:, 2], color=tuple(col))
        ax2_2.set_title('Clusters: %d' % (cluster_number))
    else:
        ax2_2 = fig.add_axes([0.75, 0.54, 0.2, 0.32])
        err_distribution = ax2_2.contour(window_end*delta, window_start*delta, err.reshape([window_start.size, window_end.size]), colors='k')
        colors = [plt.cm.Spectral(each) for each in np.linspace(0.5, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 0.1]
            elif k == -2:
                col = [0.8, 0.1, 0.1, 0.2]
            elif k == maxlabel:
                col = [0.8, 0.1, 0.1, 0.5]
            else:
                col = list(col)
                col[3] = 0.3
            data_p = sidata[labels==k]
            ax2_2.scatter(data_p[:, 0], data_p[:, 1], color=tuple(col))
        ax2_2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2_2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2_2.yaxis.set_ticks_position('right')
        ax2_2.axhline(start_opt*delta, color='crimson')
        ax2_2.axvline(end_opt*delta, color='crimson')
        ax2_2.clabel(err_distribution, inline=True)
        ax2_2.set_title('Error')

    ax3 = fig.add_axes([0.05, 0.08, 0.425, 0.38]) # particle motion
    mid = (np.average(tdata_window), np.average(-qdata_window))
    m = max([max(np.absolute(-qdata_window)), max(np.absolute(tdata_window))])
    qdata_window_filtered = lowpass(qdata_window, freq=0.05, df=1/delta, corners=2, zerophase=True)
    tdata_window_filtered = lowpass(tdata_window, freq=0.05, df=1/delta, corners=2, zerophase=True)
    ax3.plot(tdata_window_filtered,-qdata_window_filtered, color='grey')
    ax3.scatter(tdata_window_filtered[0], -qdata_window_filtered[0], color='grey')
    ax3.axline(mid, slope=np.tan((90-paz_opt)*np.pi/180), color='crimson', label='calc_pd')
    if faz is not None:
        az_diff = faz - paz_opt
        while az_diff <= -90:
            az_diff += 180
        while az_diff > 90:
            az_diff -= 180
        ax3.axline(mid, slope=np.tan((90-faz)*np.pi/180), color='mediumpurple', label='theo_pd')
        ax3.set_title('Polarization direction: %.2f\u00b0 (%.2f\u00b0)' % (paz_opt, az_diff))
    else:
        ax3.set_title('Polarization direction: %.2f\u00b0' % paz_opt)
    ax3.set_xlim(-2.148*m, 2.148*m)
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
    ax4.text(0.01*pdata_norm.size, -0.98, str(round(start_opt*delta, 1)),ha='left', va='bottom', color='darkgrey')
    ax4.text(0.99*pdata_norm.size, -0.98, str(round(end_opt*delta, 1)),ha='right', va='bottom', color='darkgrey')
    ax4.set_title('Cross-correlation: %.2f (shift=%.2fs)' % (ccvalue_opt, ccshift_opt*delta))
    ax4.set_ylim(-1, 1)
    ax4.legend()

    ax5 = fig.add_axes([0.72, 0.015, 0.05, 0.04])
    btn1 = Button(ax5, 'wait', color='darkgrey')
    btn1.on_clicked(wait)
    ax6 = fig.add_axes([0.78, 0.015, 0.05, 0.04])
    btn2 = Button(ax6, 'quit', color='darkgrey')
    btn2.on_clicked(quit)
    ax7 = fig.add_axes([0.84, 0.015, 0.05, 0.04])
    btn3 = Button(ax7, 'abandon', color='darkgrey')
    btn3.on_clicked(abandon)
    ax8 = fig.add_axes([0.9, 0.015, 0.05, 0.04])
    btn4 = Button(ax8, 'record', color='darkgrey')
    btn4.on_clicked(record)
    plt.show()

def rotate_data(y, x, theta):
    '''Rotate theta(degrees)'''
    theta = math.radians(theta)
    y_prime = y*np.cos(theta) + x*np.sin(theta)
    x_prime = -y*np.sin(theta) + x*np.cos(theta)

    return y_prime, x_prime

def standard(data):
    si_sort = np.sort(data[:, 2])
    si_sort = si[int(si_sort.size*0.2) : int(si_sort.size*0.8)]
    si_range = np.max(data[:, 2])-np.min(data[:, 2])
    if si_range > 1:
        si_range = 1
    if si_range < 0.5:
        si_range = 0.5
    scale_start = (window_start2-window_start1) * delta / si_range
    scale_end = (window_end2-window_end1) * delta / si_range
    data_scaled = data.copy()
    data_scaled[:, 0] = data_scaled[:, 0] / scale_start
    data_scaled[:, 1] = data_scaled[:, 1] / scale_end
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
    if os.path.isdir(args.filespath) is False:
        st = read(args.filespath+'*')
        add_header()
        if args.preprocess: preprocess()
        ldata, qdata, tdata = prepare()
        M = moment_tensor()
        process()
        # try:
        #     process()
        # except:
        #     print('------'+station_code+'------')
    else:
        dir = args.filespath
        bhzlist = fnmatch.filter(os.listdir(dir), '*BHZ')
        stalist = [bhz[:6] for bhz in bhzlist]
        need_moment = True
        wait_list = []
        for stnm in stalist:
            st = read(dir+stnm+'*')
            add_header()
            if args.preprocess: preprocess()
            ldata, qdata, tdata = prepare()
            if need_moment is True:
                M = moment_tensor()
                need_moment = False
            try:
                process()
            except:
                print('------'+station_code+'------')
        for stnm in wait_list:
            st = read(dir+stnm+'*')
            add_header()
            if args.preprocess: preprocess()
            ldata, qdata, tdata = prepare()
            args.autopick = False
            try:
                process()
            except:
                print('------'+station_code+'------')
        # f = open('./comp.lst', 'a+')
        # f.write(dir.split('/')[2])
        # f.close()