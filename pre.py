from obspy.io.xseed import Parser
from obspy import read, read_inventory
from obspy.core.utcdatetime import UTCDateTime
import os
import fnmatch
import numpy as np
from distaz import DistAz
from obspy.taup import TauPyModel

dir = '/Users/youyu/projects/20230807171945.SEED/2013/'
dir_x = '/Users/youyu/projects/2/data/'
for dir_m in os.listdir(dir):
    if os.path.isdir(dir+dir_m) is False:
        continue
    for file in os.listdir(dir+dir_m):
        try:
            sp = Parser(dir+dir_m+'/'+file)
            os.makedirs(dir_x+file.split('.')[0])
        except:
            continue
        sp.write_xseed(dir_x+file.split('.')[0]+'/inv.xml')
        st = read(dir+dir_m+'/'+file)
        for tr in st:
            tr.write(dir_x+file.split('.')[0]+'/'+tr.id, format='sac')

sta, stla, stlo = np.loadtxt('../Station-CN.txt', dtype={'names': ('sta', 'stla', 'stlo'), 'formats': ('U10', 'f4', 'f4')}, unpack=True)
date, time, evla, evlo, evdp = np.loadtxt('../cat.lst', usecols=(0, 1, 2, 3, 4), dtype={'names': ('date', 'time', 'evla', 'evlo', 'evdp'), 'formats': ('U10', 'U12', 'f4', 'f4', 'f4')}, unpack=True)
for dir_e in os.listdir(dir_x):
    if os.path.isdir(dir_x+dir_e) is False:
        continue
    origin = str(int(dir_e)/10)
    origin = UTCDateTime(origin)
    for i in range(date.size):
        if abs(origin-UTCDateTime(date[i]+' '+time[i])) <= 5:
            # origin = UTCDateTime(date[i]+' '+time[i])
            break
    bhzlist = fnmatch.filter(os.listdir(dir_x+dir_e), '*BHZ')
    stalist = [bhz[:6] for bhz in bhzlist]
    for stnm in stalist:
        st = read(dir_x+dir_e+'/'+stnm+'*')
        stindex = np.where(sta==stnm)
        if len(stindex[0]) == 0:
            os.system('rm '+dir_x+dir_e+'/'+stnm+'*')
            continue
        results = DistAz(stla[stindex[0]], stlo[stindex[0]], evla[i], evlo[i])
        gcarc = results.getDelta()
        model = TauPyModel('iasp91')
        arrivals = model.get_ray_paths(evdp[i], gcarc, ['S'])
        if len(arrivals) == 0:
            os.system('rm '+dir_x+dir_e+'/'+stnm+'*')
            continue
        for tr in st:
            hdr = tr.stats.sac
            hdr.o = origin - st[0].stats.starttime
            hdr.a = hdr.o + arrivals[0].time
            hdr.stla, hdr.stlo = stla[stindex[0]], stlo[stindex[0]]
            hdr.evla, hdr.evlo, hdr.evdp = evla[i], evlo[i], evdp[i]
            tr.trim(st[0].stats.starttime+hdr.a-60, st[0].stats.starttime+hdr.a+120)
            tr.write(dir_x+dir_e+'/'+tr.id, format='sac')