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
# Bokeh imports
import bokeh
import bokeh.plotting
import argparse
import sys
import h5py
__version__ = "0.0.1"

def read_Nerti_beam(fitter_file):
    with h5py.File(fitter_file,'r') as fp:
        BeamCodes = fp['/BeamCodes'][:]
        Ne = fp['/FittedParams/Ne'][:]
        Altitude = fp['/FittedParams/Altitude'][:]
        UnixTime = fp['/Time/UnixTime'][:]
    return BeamCodes,UnixTime,Altitude,Ne

parser = argparse.ArgumentParser(description='get_rti_blocks',
                    epilog="Usage :"\
              "$ bokeh serve get_rti_blocks.py --port XXXX --args fitter_file")
parser.add_argument('fitter_file', help='Fitted file.')
parser.add_argument('--beam', type=int, default=0, help='Beam index to plot first.')
parser.add_argument('--maxbeams', type=int, default=0, help='Maximum number of beams to load.')

args = parser.parse_args()

fitter_file = args.fitter_file
print(fitter_file)
bmi=args.beam
maxbeams = args.maxbeams
#bmi=0
#maxbeams = 0

print('File to use:',fitter_file)
print('Beam to plot first:',bmi)
try:
    BeamCodes,UnixTime,Altitude,Ne = read_Nerti_beam(fitter_file)
except Exception as e:
    print(e)
    sys.exit()
print(f"BeamCodes.shape : {BeamCodes.shape}")

if maxbeams > 0:
    BeamCodes = BeamCodes[:maxbeams]

dirname = os.path.dirname(fitter_file)
outfolder = os.path.join(dirname,'unblocked')
os.makedirs(outfolder, exist_ok=True)
if 'lp' in os.path.basename(fitter_file):
    block_file = os.path.join(outfolder,"block_lp.txt")
elif 'ac' in os.path.basename(fitter_file):
    block_file = os.path.join(outfolder,"block_ac.txt")


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
y=[]
dh=[]
image=[]

def nangaps(UTime,Ne):
    t_diffs = np.diff(UTime[:,0])
    t_res = np.nanmedian(t_diffs)
    ts2insert = np.array([int(max(0,x)) for x in t_diffs//t_res-1])
    if any(ts2insert>0):
        indices2ins = np.concatenate([np.array(x*[i+1]) for i,x in enumerate(ts2insert) if x>0])
        return np.insert(Ne,indices2ins,np.nan, axis=0)
    else:
        return Ne

for i in range(BeamCodes.shape[0]):
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
    y.append(valid_altitudes[0])
    dh.append(valid_altitudes[-1] - valid_altitudes[0])
    image.append(valid_Ne.T)

source_rtis = bokeh.models.ColumnDataSource(data=dict(
                            image=image,
                            x=x,
                            y=y,
                            dw=dw,
                            dh=dh
                            ))
source_rti = bokeh.models.ColumnDataSource(data=dict(
                            image=image[bmi:bmi+1],
                            x=x[bmi:bmi+1],
                            y=y[bmi:bmi+1],
                            dw=dw[bmi:bmi+1],
                            dh=dh[bmi:bmi+1]
                            ))
vmin = "10"
vmax = "12"
slider_vmin = 8
slider_vmax = 15

plot_width =850
plot_height=500
color_mapper = bokeh.models.mappers.LogColorMapper(palette="Viridis256", low=10**float(vmin), high=10**float(vmax))
p = bokeh.plotting.figure(plot_width = plot_width, plot_height=plot_height,
                          x_range=[0, dw[0]], y_range=[y[0], y[0]+dh[0]])
im = p.image(source=source_rti, color_mapper=color_mapper)

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
button1 = bokeh.models.Button(label="Delete selected row(s)", button_type="success")
button2 = bokeh.models.Button(label="erase all", button_type="success")
button3 = bokeh.models.Button(label="unselect", button_type="success")
button4 = bokeh.models.Button(label="Copy to all beams", button_type="success")
button5 = bokeh.models.Button(label="Save data", button_type="success")

box_edit_tool1 = bokeh.models.BoxEditTool(renderers=[r1])
p.add_tools(box_edit_tool1)
p.toolbar.active_drag = box_edit_tool1

def save_data():
    print(f"Saving: {block_file}")
    with open(block_file,'w')as fp:
        for bcode,vals in block_dict.items():
            for x0,x1,y0,y1 in zip(vals['x0'],vals['x1'],vals['y0'],vals['y1']):
                fp.write(f"{bcode} {x0} {x1} {y0} {y1}\n")

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

button1.on_click(b1delete_selected)
button2.on_click(b2erase_all)
button3.on_click(b3unselect)
button4.on_click(b4copy2allbeams)
button5.on_click(b5savedata)
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
                      for i in range(BeamCodes.shape[0])],width=int(cwidth/4.))
button_nextbm = bokeh.models.Button(label="next beam", button_type="success",width=int(cwidth/4.))
if bmi ==0:
    prev_disabled = True
else:
    prev_disabled = False
button_prevbm = bokeh.models.Button(label="prev beam", button_type="success",width=int(cwidth/4.), 
        disabled=prev_disabled)

callback_changebmi = bokeh.models.CustomJS(args=dict(
            source_rti=source_rti,source_rtis=source_rtis,select_bmi=select_bmi
        ), code = """
        var bmi = parseInt(select_bmi.value);
        source_rti.data['image'][0] = source_rtis.data['image'][bmi]
        source_rti.data['x'][0] = source_rtis.data['x'][bmi]
        source_rti.data['y'][0] = source_rtis.data['y'][bmi]
        source_rti.data['dw'][0] = source_rtis.data['dw'][bmi]
        source_rti.data['dh'][0] = source_rtis.data['dh'][bmi]
        source_rti.change.emit();
        """)
select_bmi.js_on_change('value',callback_changebmi)
select_bmi.on_change('value',updatebmi)
jscallback_next = bokeh.models.CustomJS(args=dict(
    select_bmi=select_bmi,button_prevbm=button_prevbm,button_nextbm=button_nextbm), code =  """
        console.log('next beam pressed');
        // console.log(select_bmi);
        var bmi_orig = select_bmi.value;
        console.log('bmi was :' + bmi_orig);
        var next_val_int = parseInt(bmi_orig) + 1;
        var next_val = next_val_int.toString();
        var last_index = parseInt(select_bmi.options.at(-1)[0]);
        if (next_val_int == last_index) {
            button_nextbm['disabled'] = true;
        }
        if (next_val_int > 0) {
            console.log(button_prevbm);
            button_prevbm['disabled'] = false;
        }
        select_bmi.value = next_val;

    """)
button_nextbm.js_on_click(jscallback_next)
jscallback_prev = bokeh.models.CustomJS(args=dict(
    select_bmi=select_bmi,button_prevbm=button_prevbm,button_nextbm=button_nextbm), code =  """
        console.log('prev beam pressed');
        var bmi_orig = select_bmi.value;
        var next_val_int = parseInt(bmi_orig) - 1;
        var next_val = next_val_int.toString();
        var last_index = parseInt(select_bmi.options.at(-1)[0]);
        if (next_val_int == 0 ) {
            button_prevbm['disabled'] = true;
        }
        if (next_val_int < last_index) {
            button_nextbm['disabled'] = false;
        }
        select_bmi.value = next_val;

    """)
button_prevbm.js_on_click(jscallback_prev)


buttons = bokeh.layouts.column(button1,button2,button3,button4,button5)
blocks_ctrl = bokeh.layouts.row(data_table,buttons)
data_ctrl = bokeh.layouts.row(
        bokeh.layouts.column(select_bmi,button_prevbm,button_nextbm,input_vmin,input_vmax,slider_vmin_vmax)
        ,p
    )
col = bokeh.layouts.column(data_ctrl,blocks_ctrl)

bokeh.io.curdoc().add_root(col)
