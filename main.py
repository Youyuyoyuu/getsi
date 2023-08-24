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
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client
from obspy import read, read_inventory
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.trigger import recursive_sta_lta, plot_trigger, trigger_onset

def autopick_time_window(stream, faz):
    qdata, tdata = stream[1].data, stream[2].data
    if faz != 0:
        pdata, odata = rotate_data(-qdata, tdata, faz)
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
        opt_w = np.unravel_index(snr.argmax(), snr.shape)[0]
        window_start2 = window[opt_w][0] * delta
        window_start1 = window_start2 - args.autopick
        window_end1 = window[opt_w][1] * delta
        window_end2 = window_end1 + args.autopick

        return window_start1, window_start2, window_end1, window_end2       

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filesname', help='files\' name without extension')
    parser.add_argument('-a', '--autopick', help='automatically pick time window (a timespan is needed)', type=float)
    parser.add_argument('-f', '--filter', help='default bandpass filter (0.02-0.2Hz)', action='store_true')
    parser.add_argument('-p', '--preprocess', help='detrend, taper and remove instrument response', action='store_true')
    parser.add_argument('--force', help='ingnore previous execution and force to run the code once', action='store_true')
    args = parser.parse_args()

    return args

def calculate_si(pdata, odata):
    '''Calculate splitting intensity and cross-correlation with time shift.'''
    global derp_norm, pdata_norm, odata_norm, err, ccvalue, ccshift
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
    err = np.sqrt((np.dot(odata_norm, odata_norm.T)-0.25*sq_rmsdr*si**2)/npoint)
    shift_limit = int(2/delta)
    cc = correlate(derp_norm, odata_norm, shift_limit)
    ccshift, ccvalue = xcorr_max(cc)

    return si, err, ccvalue

def flinn(stream, noise_thres=0):
    '''Manually changed from obspy's source code.'''
    mask = (stream[0][:] ** 2 + stream[1][:] ** 2 + stream[2][:] ** 2) > noise_thres
    x = np.zeros((3, mask.sum()), dtype=np.float64)
    # East for x0, here changed to T
    x[0, :] = stream[2][mask] #
    # North for x1, here changed to R or -Q
    x[1, :] = -stream[1][mask]
    # Z for x2, here changed to L
    x[2, :] = stream[0][mask] #

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

def grid_search():
    global window_start, window_end, si, eva, paz, opt_i, opt_j, st_window
    window_start = np.arange(window_start1, window_start2, 0.2)
    window_end = np.arange(window_end1, window_end2, 0.2)
    si = np.zeros((window_start.size, window_end.size))
    # err = np.zeros((window_start.size, window_end.size))
    eva = np.zeros((window_start.size, window_end.size))
    for i in range(window_start.size):
        for j in range(window_end.size):
            st_window = st_filtered.slice(b_utc+window_start[i], b_utc+window_end[j])
            paz = flinn(st_window)
            pdata_window, odata_window = rotate_data(-st_window[1].data, st_window[2].data, paz)
            si[i][j], err, ccvalue = calculate_si(pdata_window, odata_window)
            eva[i][j] = err#
    opt_i, opt_j = np.unravel_index(eva.argmin(), eva.shape)
    st_window = st_filtered.slice(b_utc+window_start[opt_i], b_utc+window_end[opt_j])
    paz = flinn(st_window)
    pdata_window, odata_window = rotate_data(-st_window[1].data, st_window[2].data, paz)
    result = calculate_si(pdata_window, odata_window)

    return opt_i, opt_j

def moment_tensor():
    '''Search the moment tensor of a certain event.
    If not exist, return None.
    '''
    evt_search = event_time[:16]
    evt_search = evt_search.replace('-', '/')
    evt_search = evt_search.replace('T', ' ')
    fm = open('./jan76_dec20.ndk', 'r')
    text = np.array(fm.readlines())
    mo = fnmatch.filter(text, '*'+evt_search+'*')
    if len(mo) < 1: M is None
    index = np.where(text==mo[0])[0][0]
    # event_info = text[index].split()
    # date, time, lat, lon, dep, mag = event_info[1:7]
    moment = text[index+3].split()
    expo, M= moment[0], moment[1::2]
    M = [float(m) for m in M]

    return M

def pick_time_window(stream, faz):
    '''The input stream must be in LQT order.
    If theoretical polarization direction is not provided,
    only Q and T's waveforms will be displayed.
    '''
    qdata, tdata = stream[1].data, stream[2].data
    if faz != 0:
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

    npts = stream[0].stats.npts

    fig = plt.figure(figsize=[16, 6])
    ax0 = fig.add_axes([0.37, 0.025, 0.05, 0.05], frame_on=False, xticks=[], yticks=[])
    ax0.text(0.5, 0.5, pair_name)
    ax1 = fig.add_axes([0.03, 0.15, 0.95, 0.80])
    x = np.arange(npts)
    if faz == 0:
        ax1.plot(x, qdata, color='crimson', label='Q')
        ax1.plot(x, tdata, color='dimgrey', linestyle='dashed', label='T')
    else:
        ax1.plot(x, pdata, color='firebrick', label='P')
        ax1.plot(x, odata, color='midnightblue', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax1.axvline(x=30/delta, color='darkgrey', linestyle='dotted')
    ax1.set_xlim(0, math.floor(90/delta))
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
    window_start1, window_start2, window_end1, window_end2 = round(timepoint[0]*delta, 2), round(timepoint[1]*delta, 2), round(timepoint[2]*delta, 2), round(timepoint[3]*delta, 2)

    return window_start1, window_start2, window_end1, window_end2

def prepare():
    '''Rotate from original ZNE to LQT components, 
    meanwhile write incident and take-off angle into new files.
    '''
    model = TauPyModel('iasp91')
    hdr = st[0].stats.sac
    arrivals = model.get_ray_paths(hdr.evdp/1000, hdr.gcarc, ['S'])
    inc = arrivals[0].incident_angle
    tak = arrivals[0].takeoff_angle
    baz = hdr.baz
    try:
        st.rotate('ZNE->LQT', baz, inc) # st[0] for T, st[1] for Q, st[2] for L
        st[2].stats.sac.cmpinc = inc
        st[2].stats.sac.user0 = tak
        for tr in st:
            tr.write('./cache/temp/'+pair_name+'.'+tr.stats.channel, 'SAC')
    except:
        raise ValueError(pair_name+': Failed to rotate to LQT channels.')

def preprocess():
    '''Preprocess the raw data.
    If there is no response for a certain instrument 
    either in a collection or in a single file, 
    search and write down one for this instrument individually.
    '''
    st.detrend('constant')
    st.detrend('linear')
    st.taper(max_percentage=0.05)
    try:
        try:
            inv = read_inventory('../inv.xml')
        except:
            inv = read_inventory('./cache/resp/'+station_code+'.xml')
    except:
        client = Client('IRIS')
        channel = st[0].stats.channel[:2]+'*'
        starttime = st[0].stats.starttime
        endtime = st[0].stats.endtime
        if location is None:
            location_search = '--'
        client.get_stations(starttime=starttime, endtime=endtime,
                            network=net, station=station, location=location_search, channel=channel,
                            level='response', filename='./cache/resp/'+station_code+'.xml', format='xml')
        inv = read_inventory('./cache/resp/'+station_code+'.xml')
    st.remove_response(inventory=inv)

def result_plot():
    '''Show up the final results in a figure.'''
    fig = plt.figure(figsize=[16, 10])

    ax0 = fig.add_axes([0.05, 0.54, 0.425, 0.38], frame_on=False, xticks=[], yticks=[]) # information
    text_content = 'Event: '+event_time+'\n'\
                    +'Station: '+station_code+'\n'\
                    +'SI = '+str(round(si[opt_i][opt_j], 2))+'\u00B1'+str(round(err, 2))
    ax0.text(0.3, 1.0, text_content, bbox=dict(facecolor='white', linewidth=0.5),ha='left', va='top',fontsize=12, linespacing=1.8)

    ax1 = fig.add_axes([0.05, 0.54, 0.425, 0.28]) # waveform
    pdata, odata = rotate_data(-st_filtered[1].data, st_filtered[2].data, paz)
    x = np.arange(pdata.size)
    ax1.plot(x, pdata, color='crimson', label='P')
    ax1.plot(x, odata, color='dimgrey', linestyle='dashed', label='O')
    ax1.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax1.axvline(x=30/delta, color='darkgrey', linestyle='dotted')
    rect1_1 = pch.Rectangle((window_start1/delta, -1), (window_start2-window_start1)/delta, 2, color='teal', alpha=0.05)
    ax1.add_patch(rect1_1)
    rect2 = pch.Rectangle((window_end1/delta, -1), (window_end2-window_end1)/delta, 2, color='darkslateblue', alpha=0.05)
    ax1.add_patch(rect2)
    ax1.axvline(x=window_start[opt_i]/delta, color='teal')
    ax1.axvline(x=window_end[opt_j]/delta, color='darkslateblue')
    ax1.set_xlim(0, math.floor(90/delta))
    ax1.legend()
    ax1.xaxis.set_major_locator(MultipleLocator(20/delta))
    ax1.xaxis.set_minor_locator(MultipleLocator(5/delta))
    xlabel_original = np.arange(0, pdata.size, 20/delta)
    xlabel_changed = xlabel_original*delta
    ax1.set_xticks(xlabel_original, xlabel_changed)

    ax2_1 = fig.add_axes([0.525, 0.54, 0.2, 0.32]) # grid search
    si_distribution = ax2_1.contour(window_end, window_start, si, colors='k')
    ax2_1.xaxis.set_major_locator(MultipleLocator(2))
    ax2_1.xaxis.set_minor_locator(MultipleLocator(1))
    ax2_1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2_1.axhline(window_start[opt_i], color='black')
    ax2_1.axvline(window_end[opt_j], color='black')
    ax2_1.clabel(si_distribution, inline=True)
    ax2_1.set_title('SI')
    ax2_2 = fig.add_axes([0.75, 0.54, 0.2, 0.32])
    ax2_2.axhline(window_start[opt_i], color='black')
    ax2_2.axvline(window_end[opt_j], color='black')
    eva_distribution = ax2_2.contour(window_end, window_start, eva, colors='k')
    ax2_2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2_2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2_2.clabel(eva_distribution, inline=True)
    ax2_2.set_title('Error/cc')

    ax3 = fig.add_axes([0.05, 0.08, 0.425, 0.38]) # particle motion
    mid = (np.average(st_window[2].data), np.average(-st_window[1].data))
    m = max([max(np.absolute(-st_window[1].data)), max(np.absolute(st_window[2].data))])
    ax3.plot(st_window[2].data,-st_window[1].data, color='grey')
    ax3.scatter(st_window[2].data[0], -st_window[1].data[0], color='grey')
    ax3.axline(mid, slope=np.tan((90-paz)*np.pi/180), color='crimson', label='calc_pd')
    ax3.axline(mid, slope=np.tan((90-faz)*np.pi/180), color='mediumpurple', label='theo_pd')
    ax3.set_title('Polarization direction: %.2f\u00b0 (%.2f\u00b0)' % (paz, faz))
    ax3.set_xlim(-1.2*m, 1.2*m)
    ax3.set_ylim(-1.2*m, 1.2*m)
    ax3.set_xlabel('T')
    ax3.set_ylabel('-Q')
    ax3.legend()

    ax4 = fig.add_axes([0.525, 0.08, 0.425, 0.38]) # time window
    ax4.plot(np.arange(derp_norm.size), derp_norm, color='brown', label='derP')
    ax4.plot(np.arange(odata_norm.size)+ccshift, odata_norm, color='black', ls='dashed', label='O')
    ax4.plot(np.arange(pdata_norm.size), pdata_norm, color='crimson', label='P', linestyle=(0, (1, 5)))
    ax4.axhline(y=0, color='darkgrey', linestyle='dotted')
    ax4.axvline(x=0, color='darkgrey', linestyle='dotted')
    ax4.axvline(x=pdata_norm.size, color='darkgrey', linestyle='dotted')
    ax4.xaxis.set_major_locator(MultipleLocator(1/delta))
    ax4.set_xticklabels([])
    ax4.text(0, -1, str(round(window_start[opt_i], 2)),ha='left', va='bottom', color='darkgrey')
    ax4.text(pdata_norm.size, -1, str(round(window_end[opt_j], 2)),ha='right', va='bottom', color='darkgrey')
    ax4.set_title('Cross-correlation: %.2f (shift=%.2fs)' % (ccvalue, ccshift*delta))
    ax4.set_ylim(-1, 1)
    ax4.legend()

    plt.show()

def rms(array):
    return np.sqrt(np.sum(array**2)/len(array))

def rotate_data(y, x, theta):
    '''Rotate theta(degrees)'''
    theta = math.radians(theta)
    y_prime = y*np.cos(theta) + x*np.sin(theta)
    x_prime = -y*np.sin(theta) + x*np.cos(theta)

    return y_prime, x_prime

if __name__ == '__main__':
    args = get_arguments()
    # args =  # for debugging
    st = read('../'+args.filesname+'.*Z')
    net = st[0].stats.network
    station = st[0].stats.station
    location = st[0].stats.location
    station_code = net+'.'+station+'.'+location
    event_time = (st[0].stats.starttime+st[0].stats.sac.o).format_iris_web_service()
    pair_name = event_time+'-'+station_code
    filelist = fnmatch.filter(os.listdir('./cache/temp/'), pair_name+'.*')
    if len(filelist)==0 or args.force:
        st = read('../'+args.filesname+'.*')
        if args.preprocess:
            preprocess()
        prepare()
    st_lqt = read('./cache/temp/'+pair_name+'.*')
    M = moment_tensor()
    try:
        az, tak, delta, b_utc = st_lqt[0].stats.sac.az, st_lqt[0].stats.sac.user0, st_lqt[0].stats.delta, st_lqt[0].stats.starttime
    except:
        raise ValueError(pair_name+': Lack header infomations.')
    st_filtered = st_lqt.copy()
    if args.filter:
        st_filtered.filter('bandpass', freqmin=0.02, freqmax=0.2, corners=2, zerophase=True)
    if args.autopick:
        if M is None:
            window_start1, window_start2, window_end1, window_end2 = autopick_time_window(st_filtered, 0)
        else:
            faz = focal_polarization(M, tak, az)
            window_start1, window_start2, window_end1, window_end2 = autopick_time_window(st_filtered, faz)
    else:
        if M is None:
            window_start1, window_start2, window_end1, window_end2 = pick_time_window(st_filtered, 0)
        else:
            faz = focal_polarization(M, tak, az)
            window_start1, window_start2, window_end1, window_end2 = pick_time_window(st_filtered, faz)
    grid_search()
    result_plot()