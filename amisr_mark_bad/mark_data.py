"""
File: mark_data.py

Author: Pablo M. Reyes
Initial implementation: 2/2/2022
Description: Tool to mark bad data on AMISR RTIs

Usage: 
$ bokeh serve get_rti_blocks.py --port 5007
Written for bokeh 2.4.2 in a py38 environment

"""

# External imports
import numpy as np
import os
import shutil
import filecmp
import glob
import signal
from functools import partial

# Bokeh imports
import bokeh
import bokeh.plotting

import datetime
import argparse
import sys
import h5py
import colorcet as cc

__version__ = "0.0.2"
def read_Nerti_beam(fitter_file,param_group,param2plot):
    parampath = '/'+param_group+'/'+param2plot
    param_hts = '/'+param_group+'/Altitude'
    print(f'Parameter to plot: {parampath}')
    with h5py.File(fitter_file,'r') as fp:
        BeamCodes = fp['/BeamCodes'][:]
        param = fp[parampath][:]
        recs,nbeams,oldalts = param.shape
        Altitude = fp[param_hts][:]
        print("param",param.shape)
        print("Altitude",Altitude.shape)
        nbeams,nhts0 = Altitude.shape
        newhts = []
        maxalts = 0
        for bmi in range(nbeams):
            hmindiff = np.nanmin(np.diff(Altitude[bmi,:]))
            validAlts = np.where(np.isfinite(Altitude[bmi,:]))[0]
            valid_i0 = validAlts[0]
            valid_i1 = validAlts[-1]
            newhts_tmp = np.arange(Altitude[bmi,valid_i0],Altitude[bmi,valid_i1]+hmindiff,hmindiff)
            if len(newhts_tmp)>maxalts:
                maxalts = len(newhts_tmp)
            newhts.append(newhts_tmp)
        param_interp = np.empty((recs,nbeams,maxalts))*np.nan
        Alts_interp = np.empty((nbeams,maxalts)) * np.nan
        for bmi in range(nbeams):
            htsi = len(newhts[bmi])
            Alts_interp[bmi,:htsi] = newhts[bmi]
            for rec0 in range(recs):
                for hti, h0 in enumerate(newhts[bmi]):
                    validAlts = np.where(np.isfinite(Altitude[bmi,:]))[0]
                    orig_hi = np.where(h0 >= Altitude[bmi][validAlts])[0][-1]
                    param_interp[rec0,bmi,hti] = param[rec0,bmi,orig_hi]
        UnixTime = fp['/Time/UnixTime'][:]
        # fix short start and end
        tdiff = np.nanmedian(UnixTime[:,1]-UnixTime[:,0])
        UnixTime[-1,1] = UnixTime[-1,0] + tdiff
        UnixTime[0,0] = UnixTime[0,1] -  tdiff
    return BeamCodes,UnixTime,Alts_interp,param_interp

parser = argparse.ArgumentParser(description='get_rti_blocks',
                    epilog="Usage :"\
              "$ bokeh serve get_rti_blocks.py --port XXXX --args fitter_file")
parser.add_argument('fitter_file', help='Fitted file.')
parser.add_argument('--beam', type=int, default=0, help='Beam index to plot first.')
parser.add_argument('--maxbeams', type=int, default=0, help='Maximum number of beams to load.')
parser.add_argument('--param_group', action='store', type=str, default='FittedParams', help='Group of the parameter to plot.')
parser.add_argument('--param2plot', action='store', type=str, default='Ne', help='Group of the parameter to plot.')
args = parser.parse_args()

fitter_file = args.fitter_file
print(fitter_file)
first_bmi=args.beam
bmi = first_bmi
maxbeams = args.maxbeams
param_group = args.param_group
param2plot = args.param2plot
#bmi=0
#maxbeams = 0

print('File to use:',fitter_file)
print('Beam to plot first:',bmi)
try:
    BeamCodes,UnixTime,Altitude,Ne = read_Nerti_beam(fitter_file,param_group,param2plot)
except Exception as e:
    print(e)
    print("Trying --param_group NeFromPower --param2plot Ne_NoTr")
    param_group = "NeFromPower"
    param2plot = "Ne_NoTr"
    try:
        BeamCodes,UnixTime,Altitude,Ne = read_Nerti_beam(fitter_file,param_group,param2plot)
    except Exception as e:
        print(e)
        sys.exit()
print(f"BeamCodes.shape : {BeamCodes.shape}")

bmis2show = list(range(BeamCodes.shape[0]))
if maxbeams > 0:
    bmis2show =  bmis2show[bmi:][:maxbeams]

def get_sintg(fname):
    bname = os.path.basename(fname)
    minloc = bname.find('min')
    print(f"Location of min: minloc={minloc}")
    if minloc>=0:
        numloc = (bname[:minloc]).rfind("_")
        return "_"+bname[numloc+1:minloc]+"min"
    else:
        print("No min found")
        secloc = bname.find('sec')
        print(f"Location of sec: secloc={secloc}")
        if secloc>=0:
            numloc = (bname[:secloc]).rfind("_")
            return "_"+bname[numloc+1:secloc]+"sec"
        else:
            return ""
dirname = os.path.dirname(fitter_file)
outfolder = os.path.join(dirname,'unblocked')
trimfolder = os.path.join(dirname,'untrimmed')
sintg = get_sintg(fitter_file)
if 'lp' in os.path.basename(fitter_file).lower():
    block_file = os.path.join(outfolder,f"block_lp{sintg}.txt")
    trim_file = os.path.join(trimfolder,f"trim_lp{sintg}.txt")
elif 'ac' in os.path.basename(fitter_file).lower():
    block_file = os.path.join(outfolder,f"block_ac{sintg}.txt")
    trim_file = os.path.join(trimfolder,f"trim_ac{sintg}.txt")
elif 'bc' in os.path.basename(fitter_file).lower():
    block_file = os.path.join(outfolder,f"block_bc{sintg}.txt")
    trim_file = os.path.join(trimfolder,f"trim_bc{sintg}.txt")


block_dict = {}
def reset_block_dict():
    for i,bcode in enumerate(BeamCodes[:,0]):
        print(i,bcode)
        block_dict.update({bcode:{'x0':np.array([]),
                                  'x1':np.array([]),
                                  'y0':np.array([]),
                                  'y1':np.array([])}})
reset_block_dict()
if os.path.exists(block_file):
    if os.stat(block_file).st_size > 0:
        # if size>0, continue
        backups = sorted(glob.glob(block_file + ".backup_*"))
        num_backups = len(backups)
        if num_backups == 0 or num_backups > 0 \
                and not filecmp.cmp(block_file,backups[-1]):
            new_file = block_file + f".backup_{num_backups:03d}"
            shutil.copyfile(block_file,new_file)
            print(f"Backup generated: {new_file}")

    with open(block_file,'r') as fp:
        try:
            for line in fp.readlines():
                bcode,startx,endx,startkm,endkm = np.array(line.strip().split(),
                                                           dtype=float)
                block_dict[bcode]['x0'] = np.append(block_dict[bcode]['x0'],startx)
                block_dict[bcode]['x1'] = np.append(block_dict[bcode]['x1'],endx)
                block_dict[bcode]['y0'] = np.append(block_dict[bcode]['y0'],startkm)
                block_dict[bcode]['y1'] = np.append(block_dict[bcode]['y1'],endkm)
        except:
            print(f"Errors reading {block_file}")
            reset_block_dict()


bcode = BeamCodes[bmi,0]
x0 = block_dict[bcode]['x0']
x1 = block_dict[bcode]['x1']
y0 = block_dict[bcode]['y0']
y1 = block_dict[bcode]['y1']
source_rect = bokeh.models.ColumnDataSource(data=dict(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            width=x1-x0,
            height=y1-y0,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        ))

print(f"Number of beams: {BeamCodes.shape[0]}")
x=[]
dw=[]
dt0=[]
dt1=[]
y=[]
dh=[]
image=[]
ptitle = []

def nangaps(UTime,Ne):
    t_diffs = np.diff(UTime[:,0])
    t_res = np.nanmedian(t_diffs)
    ts2insert = np.array([int(max(0,x)) for x in t_diffs//t_res-1])
    if any(ts2insert>0):
        indices2ins = np.concatenate([np.array(x*[i+1]) for i,x in enumerate(ts2insert) if x>0])
        return np.insert(Ne,indices2ins,np.nan, axis=0)
    else:
        return Ne

dt0_tmp = datetime.datetime.utcfromtimestamp(UnixTime[0,0])
dt1_tmp = datetime.datetime.utcfromtimestamp(UnixTime[-1,-1])
total_nbms = BeamCodes.shape[0]
for i in bmis2show:#range(BeamCodes.shape[0]):
    valid_hi = np.where(np.isfinite(Altitude[i,:]))[0]
    valid_altitudes = Altitude[i,valid_hi]/1e3
    valid_Ne = Ne[:,i,valid_hi]
    valid_Ne = nangaps(UnixTime,valid_Ne)

    # There is a need to fill gaps with nans
    # because the nan block routines consider x being actual
    # seconds since the start of the experiment.
    #print(np.diff(UnixTime[:,0]))
    x.append(0)
    dw.append(UnixTime[-1,-1]-UnixTime[0,0])
    dt0.append(dt0_tmp)
    dt1.append(dt1_tmp)
    y.append(valid_altitudes[0])
    dh.append(valid_altitudes[-1] - valid_altitudes[0])
    image.append(valid_Ne.T)
    tmp_title = dt0_tmp.strftime("%x %X - ") + dt1_tmp.strftime("%x %X, ") +os.path.basename(fitter_file)
    bmcode = BeamCodes[i][0]
    az = BeamCodes[i][1]
    el = BeamCodes[i][2]
    tmp_title = f"bm {i}/{total_nbms}, bmc:{int(bmcode):d} ({az:.2f} az, {el:.2f} el), {tmp_title}"
    ptitle.append(tmp_title)

bmi_index = bmis2show.index(bmi)
source_rtis = bokeh.models.ColumnDataSource(data=dict(
                            image=image,
                            x=x,
                            y=y,
                            dw=dw,
                            dh=dh,
                            ptitle=ptitle,
                            bmis2show=bmis2show
                            ))
source_rti = bokeh.models.ColumnDataSource(data=dict(
                            image=image[bmi_index:bmi_index+1],
                            x=x[bmi_index:bmi_index+1],
                            y=y[bmi_index:bmi_index+1],
                            dw=dw[bmi_index:bmi_index+1],
                            dh=dh[bmi_index:bmi_index+1]
                            ))
vmin = "10"
vmax = "12"
slider_vmin = 8
slider_vmax = 15

#plot_width =850
#plot_height=500
plot_width =1500
plot_height=500
palette = "Viridis256"
palette = cc.rainbow4

color_mapper = bokeh.models.mappers.LogColorMapper(palette=palette, low=10**float(vmin), high=10**float(vmax))
seconds_axis = bokeh.models.DataRange1d(start=0, end=dw[0],range_padding = 0)
p = bokeh.plotting.figure(width = plot_width, height=plot_height,
                          x_range=seconds_axis, y_range=[y[0], y[0]+dh[0]])
p.title.text = ptitle[bmi_index]
im = p.image(source=source_rti, color_mapper=color_mapper)
p.toolbar.active_scroll = p.select_one(bokeh.models.WheelZoomTool)
plot_width =850
plot_height=500
p.x_range.renderers = [im] # specifying the renderers for the x_range
p.xaxis.axis_label = 'seconds from experiment start (s)'
cb = bokeh.models.ColorBar(color_mapper = color_mapper, location = (5,6))
p.add_layout(cb, 'right')
#second axis
# trick: two invisible points at two diagonal opposite corners
# or   : two invisible points at two points to the left and right limits
#         put that have a common y axis point
if True:
    t_src = bokeh.models.ColumnDataSource({'x':[dt0[0],dt1[0]],'y':[y[0],y[0]]})
    # range_padding needs to coincide with the range_padding of the main figure
    p.extra_x_ranges={'t':bokeh.models.DataRange1d(range_padding = 0)}
    x2_scale = bokeh.models.LinearAxis(x_range_name='t')
    x2_scale.formatter=bokeh.models.DatetimeTickFormatter()
    x2_scale.ticker = bokeh.models.DatetimeTicker()
    x2_scale.axis_label = "time (UT)"
    p.add_layout(x2_scale,'above')
    #second axis renderer, and tell it to plot on the 'r' y_range_name
    tg = bokeh.models.Scatter(x='x',y='y')
    tt = p.add_glyph(t_src,tg,x_range_name='t')
    tt.visible = False
    #THEN, now that that renderer is instantiated, tell the y_range['r'] to follow it
    p.extra_x_ranges['t'].renderers=[tt]

cwidth = 500

r1 = p.rect(source=source_rect, color="red",fill_alpha=0.5)

columns = [
        bokeh.models.TableColumn(field='x', title='x', editor=bokeh.models.CellEditor()),
        bokeh.models.TableColumn(field='y', title='y', editor=bokeh.models.CellEditor()),
        bokeh.models.TableColumn(field='width', title='width', editor=bokeh.models.CellEditor()),
        bokeh.models.TableColumn(field='height', title='height', editor=bokeh.models.CellEditor()),
        bokeh.models.TableColumn(field='x0', title='x0'),
        bokeh.models.TableColumn(field='x1', title='x1'),
        bokeh.models.TableColumn(field='y0', title='y0'),
        bokeh.models.TableColumn(field='y1', title='y1'),
]

table_width = 550
table_height = 250
data_table = bokeh.models.DataTable(source=source_rect, columns=columns,
                                    width=table_width, height=table_height,
                                   auto_edit=True, editable=True)
cwidth2 = int(cwidth/3.)
button_delete_selected = bokeh.models.Button(label="Delete selected row(s)", button_type="success",width=cwidth2)
button_erase_all = bokeh.models.Button(label="erase all", button_type="success",width=cwidth2)
button_unselect = bokeh.models.Button(label="unselect", button_type="success",width=cwidth2)
button_copy2all = bokeh.models.Button(label="Copy to all beams", button_type="success",width=cwidth2)
button_save_nanfile = bokeh.models.Button(label="Save data", button_type="success",width=cwidth2)
button_trimbeforex1 = bokeh.models.Button(label="Trim data before max(x1)", button_type="success",width=cwidth2)
button_trimafterx0 = bokeh.models.Button(label="Trim data after min(x0)", button_type="success",width=cwidth2)
button_y0_0_y1_1000 = bokeh.models.Button(label="Selected y0 = -100,y1=1000", button_type="success",width=cwidth2)
button_y1_1000 = bokeh.models.Button(label="Selected y1 = 1000", button_type="success",width=cwidth2)
button_y1_m100 = bokeh.models.Button(label="Selected y1 = -100", button_type="success",width=cwidth2)
button_convert2ints = bokeh.models.Button(label="Convert 2 ints", button_type="success",width=cwidth2)

box_edit_tool1 = bokeh.models.BoxEditTool(renderers=[r1])
p.add_tools(box_edit_tool1)
p.toolbar.active_drag = box_edit_tool1

def save_data():
    print(f"Saving: {block_file}")
    os.makedirs(outfolder, exist_ok=True)
    with open(block_file,'w')as fp:
        for bcode,vals in block_dict.items():
            for x0,x1,y0,y1 in zip(vals['x0'],vals['x1'],vals['y0'],vals['y1']):
                fp.write(f"{bcode} {x0} {x1} {y0} {y1}\n")

def trimdata(mode="after"):
    x0_min = 1e99
    x1_max = -1e99
    for bcode,vals in block_dict.items():
        x0 = min(vals['x0'])
        if x0 < x0_min:
            x0_min = x0
        x1 = max(vals['x1'])
        if x1 > x1_max:
            x1_max = x1
    #if os.path.basename(fitter_file).find('_bc')<0:
    #    trimcommand = f"python /opt/src/cleanfit/fitted/trim_file.py {fitter_file}"
    #else:
    #    trimcommand = f"python /opt/src/cleanfit/nepow/trim_file.py {fitter_file}"
    # the fitted works for nenotr too:
    trimcommand = f"python /opt/src/cleanfit/fitted/trim_file.py {fitter_file}"

    if mode == "before":
        trimcommand += f" {x1_max} 0"
    if mode == "after":
        trimcommand += f" {x0_min}"
    print(f"saving command to {trim_file}:{trimcommand}")
    os.makedirs(trimfolder, exist_ok=True)
    with open(trim_file,'w') as fp:
        fp.write(trimcommand)

def update_rects(datadict):
    source_rect.data.update(datadict)

def update_blocks_dict():
    print("update block_dict from source_rect")
    # update block_dict based on datadict
    bmi = int(select_bmi.value) # current bmi
    bcode = BeamCodes[bmi,0]    # current beam code
    x = np.array(source_rect.data['x'], dtype=float)
    width = np.array(source_rect.data['width'], dtype=float)
    y = np.array(source_rect.data['y'], dtype=float)
    height = np.array(source_rect.data['height'], dtype=float)
    block_dict[bcode]['x0'] = x - width/2
    block_dict[bcode]['x1'] = x + width/2
    block_dict[bcode]['y0'] = y - height/2
    block_dict[bcode]['y1'] = y + height/2
    print(block_dict[bcode])

def set_selected_2val(cols="y0", vals=0):
    #for selected_i in source_rect.selected.indices:
    #    
    print(f"selected rows {cols} to {vals}")
    print((source_rect.selected.indices))
    if len(source_rect.selected.indices)>0:
        for col,val in zip(cols,vals):
            source_rect.data[col][source_rect.selected.indices] = val
        update_rects(source_rect.data)
    convert2ints()
    #source_rect.selected.indices = []

def convert2ints():
    print("converting x0,x1,y0,y1 to integers")
    print((source_rect.selected.indices))
    if len(source_rect.selected.indices)>0:
       for col in ["x0","x1","y0","y1"]:
           tmp = source_rect.data[col][source_rect.selected.indices]
           source_rect.data[col][source_rect.selected.indices] = np.round(tmp)
       update_rects(source_rect.data)
#    source_rect.selected.indices = []

def b1delete_selected(event):
    print("delete selected rows")
    print((source_rect.selected.indices))
    if len(source_rect.selected.indices)>0:
        x_rects = np.delete(source_rect.data['x'],source_rect.selected.indices) 
        y_rects = np.delete(source_rect.data['y'],source_rect.selected.indices) 
        width_rects = np.delete(source_rect.data['width'],source_rect.selected.indices) 
        height_rects = np.delete(source_rect.data['height'],source_rect.selected.indices) 
        x0_rects = np.delete(source_rect.data['x0'],source_rect.selected.indices) 
        x1_rects = np.delete(source_rect.data['x1'],source_rect.selected.indices) 
        y0_rects = np.delete(source_rect.data['y0'],source_rect.selected.indices) 
        y1_rects = np.delete(source_rect.data['y1'],source_rect.selected.indices) 
        datadict = dict(x=x_rects,y=y_rects,
                    width=width_rects,height=height_rects,
                    x0=x0_rects,x1=x1_rects,
                    y0=y0_rects,y1=y1_rects,
                    )
        update_rects(datadict)
    source_rect.selected.indices = []

def b2erase_all(event):
    print("erase all")
    x_rects = np.array([])
    y_rects = np.array([])
    width_rects = np.array([])
    height_rects = np.array([])
    x0_rects = np.array([])
    x1_rects = np.array([])
    y0_rects = np.array([])
    y1_rects = np.array([])
    datadict = dict(x=x_rects,y=y_rects,
                width=width_rects,height=height_rects,
                x0=x0_rects,x1=x1_rects,
                y0=y0_rects,y1=y1_rects,
                )
    update_rects(datadict)

def b3unselect(event):
    print("unselect all")
    source_rect.selected.indices = []

def b4copy2allbeams(event):
    print("Copy to all beams")
    bmi = int(select_bmi.value)
    bcode = BeamCodes[bmi,0]
    x0 = np.array(block_dict[bcode]['x0'])
    x1 = np.array(block_dict[bcode]['x1'])
    y0 = np.array(block_dict[bcode]['y0'])
    y1 = np.array(block_dict[bcode]['y1'])
    for x0i0,x1i0,y0i0,y1i0 in zip(x0,x1,y0,y1):
        for bcodei in BeamCodes[:,0]:
            if bcodei == bcode:
                continue
            exists = False
            for x0i1,x1i1,y0i1,y1i1 in zip(
                    block_dict[bcodei]['x0'],
                    block_dict[bcodei]['x1'],
                    block_dict[bcodei]['y0'],
                    block_dict[bcodei]['y1'],
                    ):
                # check if already exists
                if [x0i1,x1i1,y0i1,y1i1] == [x0i0,x1i0,y0i0,y1i0]:
                    exists = True
                    break
            if not exists:
                # add
                block_dict[bcodei]['x0'] = np.append(block_dict[bcodei]['x0'],x0i0)
                block_dict[bcodei]['x1'] = np.append(block_dict[bcodei]['x1'],x1i0)
                block_dict[bcodei]['y0'] = np.append(block_dict[bcodei]['y0'],y0i0)
                block_dict[bcodei]['y1'] = np.append(block_dict[bcodei]['y1'],y1i0)

def b5savedata(event):
    print("Save data.")
    save_data()
    print("done.")


# https://stackoverflow.com/questions/59196855/python-bokeh-markup-text-value-cant-update
# need a second callback
def really_stop():
    print("Now sending command to kill the app")
    sys.exit()  # Stop the server

def disable_all():
    messagediv.text = "Stoping server..."
    button_nextbm.disabled = True
    button_prevbm.disabled = True
    stop_button.disabled = True
    button_delete_selected.disabled = True
    button_erase_all.disabled = True
    button_unselect.disabled = True
    button_copy2all.disabled = True
    button_save_nanfile.disabled = True
    button_trimbeforex1.disabled = True
    button_trimafterx0.disabled = True
    button_y0_0_y1_1000.disabled = True
    button_convert2ints.disabled = True
    button_y1_1000.disabled = True
    button_y1_m100.disabled = True
    slider_vmin_vmax.disabled = True
    input_vmin.disabled = True
    input_vmax.disabled = True
    select_bmi.disabled = True


def stop_server(event):
    print("Stopping the server from the browser")
    disable_all()
    bokeh.io.curdoc().add_next_tick_callback(really_stop)

def signal_handler(signal, frame):
    print("signal.SIGINT has been received")
    #disable_all()
    really_stop()
def on_session_destroyed(session_context):
    print("on session destroyed")

signal.signal(signal.SIGINT, signal_handler)

def table_changed(source_rect,block_dict):
    bmi = int(select_bmi.value)
    bcode = BeamCodes[bmi,0]
    x0_bdict = np.array(block_dict[bcode]['x0'], dtype=float)
    x1_bdict = np.array(block_dict[bcode]['x1'], dtype=float)
    y0_bdict = np.array(block_dict[bcode]['y0'], dtype=float)
    y1_bdict = np.array(block_dict[bcode]['y1'], dtype=float)
    x0_rects = np.array(source_rect.data['x0'], dtype=float)
    x1_rects = np.array(source_rect.data['x1'], dtype=float)
    y0_rects = np.array(source_rect.data['y0'], dtype=float)
    y1_rects = np.array(source_rect.data['y1'], dtype=float)

    if len(x0_bdict) != len(x0_rects) or \
            len(x1_bdict) != len(x1_rects) or \
            len(y0_bdict) != len(y0_rects) or \
            len(y1_bdict) != len(y1_rects):
        # means that something is added or deleted so no manual change
        return 0


    if  np.any(x0_rects != x0_bdict):
        print('x0 changed')
        return 1
    if  np.any(x1_rects != x1_bdict):
        print('x1 changed')
        return 2
    if  np.any(y0_rects != y0_bdict):
        print('y0 changed')
        return 3
    if  np.any(y1_rects != y1_bdict):
        print('y1 changed')
        return 4
    return 0

def on_change_data_source(attr, old, new):
    print("updating current data squares")
    x_rects = np.array(source_rect.data['x'], dtype=float)
    y_rects = np.array(source_rect.data['y'], dtype=float)
    width_rects = np.array(source_rect.data['width'], dtype=float)
    height_rects = np.array(source_rect.data['height'], dtype=float)
    x0_rects = np.array(source_rect.data['x0'], dtype=float)
    x1_rects = np.array(source_rect.data['x1'], dtype=float)
    y0_rects = np.array(source_rect.data['y0'], dtype=float)
    y1_rects = np.array(source_rect.data['y1'], dtype=float)
    print("checking what changes")
    result = table_changed(source_rect,block_dict)
    if np.any(x0_rects != x_rects - width_rects/2) or \
           np.any(x1_rects != x_rects + width_rects/2) or \
           np.any(y0_rects != y_rects - height_rects/2) or \
           np.any(y1_rects != y_rects + height_rects/2):
        if result>0:
            source_rect.data.update({'x':(x0_rects + x1_rects)/2,
                                     'width' : x1_rects - x0_rects,
                                     'y':(y0_rects + y1_rects)/2,
                                     'height' : y1_rects - y0_rects})

        else:
            # the change was by adding or moving rects
            source_rect.data.update({
                    'x0' : x_rects - width_rects/2,
                    'x1' : x_rects + width_rects/2,
                    'y0' : y_rects - height_rects/2,
                    'y1' : y_rects + height_rects/2
                })
    update_blocks_dict()

def updatebmi(attr, old, new):
    print(f"new bmi:{select_bmi.value}")
    bmi = int(select_bmi.value)
    bcode = BeamCodes[bmi,0]
    x0 = np.array(block_dict[bcode]['x0'], dtype=float)
    x1 = np.array(block_dict[bcode]['x1'], dtype=float)
    y0 = np.array(block_dict[bcode]['y0'], dtype=float)
    y1 = np.array(block_dict[bcode]['y1'], dtype=float)
    print("type block_dict x0",type(x0))
    datadict = dict(x=(x0+x1)/2,
                    y=(y0+y1)/2,
                    width=x1-x0,
                    height=y1-y0,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1)
    update_rects(datadict)


button_delete_selected.on_click(b1delete_selected)
button_erase_all.on_click(b2erase_all)
button_unselect.on_click(b3unselect)
button_copy2all.on_click(b4copy2allbeams)
button_save_nanfile.on_click(b5savedata)
button_trimbeforex1.on_click(partial(trimdata, mode="before"))
button_trimafterx0.on_click(partial(trimdata, mode="after"))
button_y0_0_y1_1000.on_click(partial(set_selected_2val, cols=["y0","y1"], vals=[-100,1000]))
button_y1_1000.on_click(partial(set_selected_2val, cols=["y1"], vals=[1000]))
button_y1_m100.on_click(partial(set_selected_2val, cols=["y0"], vals=[-100]))
button_convert2ints.on_click(convert2ints)

source_rect.on_change('data', on_change_data_source)


JS_code_vmin_vmax = """
    var vmin = parseFloat(input_vmin.value);
    var vmax = parseFloat(input_vmax.value);
    im.glyph.color_mapper.high = Math.pow(10, vmax);
    im.glyph.color_mapper.low = Math.pow(10, vmin);
"""


input_vmin = bokeh.models.widgets.TextInput(value=vmin,title="10^vmin",width=int(cwidth/4.))
input_vmax = bokeh.models.widgets.TextInput(value=vmax,title="10^vmax",width=int(cwidth/4.))

callback_update = bokeh.models.CustomJS(code=JS_code_vmin_vmax)

input_vmax.js_on_change('value',callback_update)
input_vmin.js_on_change('value',callback_update)
callback_update.args['input_vmin'] = input_vmin
callback_update.args['input_vmax'] = input_vmax
callback_update.args['im'] = im

button_update =  bokeh.models.Button(label="Update vmin/vmax",
        width=int(cwidth/4.))
button_update.js_on_click(callback_update)

slider_vmin_vmax = bokeh.models.RangeSlider(start=slider_vmin, end=slider_vmax,
        value=(float(vmin),float(vmax)), step=.001, title="10^(vmin,vmax)",
        width=int(cwidth/4.))
callback_slider = bokeh.models.CustomJS(args=dict(
            slider_vmin_vmax=slider_vmin_vmax,
            input_vmin=input_vmin,
            input_vmax=input_vmax,
        ),
        code="""
        input_vmin.value = slider_vmin_vmax.value[0].toString();
        input_vmax.value = slider_vmin_vmax.value[1].toString();
    """)
slider_vmin_vmax.js_on_change('value',callback_slider)

select_bmi = bokeh.models.Select(title="Beam:", value=f"{bmi}",
                  options=[(f"{i}", f"bm{i}: {BeamCodes[i,0]}")
                      for i in bmis2show],width=int(cwidth/4.))
if bmi == first_bmi:
    prev_disabled = True
else:
    prev_disabled = False
if bmi == bmis2show[-1]:
    next_disabled = True
else:
    next_disabled = False
button_nextbm = bokeh.models.Button(label="next beam", button_type="success",width=int(cwidth/4.),
        disabled=next_disabled)
button_prevbm = bokeh.models.Button(label="prev beam", button_type="success",width=int(cwidth/4.),
        disabled=prev_disabled)

callback_changebmi = bokeh.models.CustomJS(args=dict(
            source_rti=source_rti,source_rtis=source_rtis,select_bmi=select_bmi,
            title=p.title
        ), code = """
        var bmi = parseInt(select_bmi.value);
        var bmis2show = source_rtis.data['bmis2show'];
        console.log('bmis2show :' + bmis2show);
        var bmi_index = bmis2show.indexOf(bmi);
        console.log('bmi_index');
        console.log(bmi_index);
        source_rti.data['image'][0] = source_rtis.data['image'][bmi_index]
        source_rti.data['x'][0] = source_rtis.data['x'][bmi_index]
        source_rti.data['y'][0] = source_rtis.data['y'][bmi_index]
        source_rti.data['dw'][0] = source_rtis.data['dw'][bmi_index]
        source_rti.data['dh'][0] = source_rtis.data['dh'][bmi_index]
        title.text = source_rtis.data['ptitle'][bmi_index]
        source_rti.change.emit();
        """)
select_bmi.js_on_change('value',callback_changebmi)
select_bmi.on_change('value',updatebmi)
jscallback_next = bokeh.models.CustomJS(args=dict(
    select_bmi=select_bmi,button_prevbm=button_prevbm,button_nextbm=button_nextbm,
                                        bmis2show=bmis2show), code =  """
        console.log('next beam pressed');
        var bmi_orig = parseInt(select_bmi.value);
        console.log('bmi was :' + bmi_orig);
        console.log(bmis2show)
        var bmi_orig_index = bmis2show.indexOf(bmi_orig);
        console.log('bmi_orig_index : '+bmi_orig_index);
        var next_val_int = parseInt(bmis2show[bmi_orig_index + 1]);
        console.log('next_val_int : '+next_val_int);
        var next_val = next_val_int.toString();
        console.log('next_val : '+ next_val);
        var last_index = parseInt(select_bmi.options.at(-1)[0]);
        if (next_val_int == last_index) {
            button_nextbm['disabled'] = true;
        }
        if (next_val_int > parseInt(bmis2show[0])) {
            console.log(button_prevbm);
            button_prevbm['disabled'] = false;
        }
        select_bmi.value = next_val;
        console.log('done with next beam pressed');
    """)
button_nextbm.js_on_click(jscallback_next)
jscallback_prev = bokeh.models.CustomJS(args=dict(
    select_bmi=select_bmi,button_prevbm=button_prevbm,button_nextbm=button_nextbm,
                                        bmis2show=bmis2show), code =  """
        console.log('prev beam pressed');
        var bmi_orig = parseInt(select_bmi.value);
        console.log('bmi was :' + bmi_orig);
        var bmi_orig_index = bmis2show.indexOf(bmi_orig);
        var next_val_int = parseInt(bmis2show[bmi_orig_index - 1]);
        var next_val = next_val_int.toString();
        console.log('next_val : '+ next_val);
        var last_index = parseInt(select_bmi.options.at(-1)[0]);
        if (next_val_int == parseInt(bmis2show[0]) ) {
            button_prevbm['disabled'] = true;
        }
        if (next_val_int < last_index) {
            button_nextbm['disabled'] = false;
        }
        select_bmi.value = next_val;

    """)
button_prevbm.js_on_click(jscallback_prev)

# Button to stop the server
stop_button = bokeh.models.Button(label="Stop Server", button_type="success",
        width=int(cwidth/4.))
stop_button.on_click(stop_server)

messagediv = bokeh.models.widgets.Div(text="Server started.",
                        width=int(cwidth/4.), height=int(cwidth/8.))

buttons_c1 = bokeh.layouts.column(button_delete_selected,
                                  button_erase_all,
                                  button_unselect,
                                  button_copy2all,
                                  button_save_nanfile)
buttons_c2 = bokeh.layouts.column(button_trimbeforex1,
                                  button_trimafterx0,
                                  button_y0_0_y1_1000,
                                  button_y1_1000,
                                  button_y1_m100,
                                  button_convert2ints)
buttons = bokeh.layouts.row(buttons_c1, buttons_c2)
blocks_ctrl = bokeh.layouts.row(data_table,buttons)
data_ctrl = bokeh.layouts.row(
        bokeh.layouts.column(select_bmi,button_prevbm,button_nextbm,
            input_vmin,input_vmax,slider_vmin_vmax,stop_button,messagediv)
        ,p
    )
col = bokeh.layouts.column(data_ctrl,blocks_ctrl)

bokeh.io.curdoc().add_root(col)
