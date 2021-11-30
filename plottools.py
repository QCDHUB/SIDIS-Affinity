import numpy as np
import pandas as pd
from functools import partial, reduce
from scipy.stats import percentileofscore
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pylab as py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def custom_label(label):
    if label == "Q2":
        label = r'\boldmath{$Q^2\; \rm(GeV^2)$}'
    if label == "qT":
        label = r'$q_T\; (GeV)$'
    if label == "W2":
        label = r'$W^2\; (GeV^2)$'
    if label == "qToverQ":
        label = r'\boldmath{$q_T/Q$}'
    if label == "qT":
        label = r'$q_T\; \rm(GeV)$'
    if label == "qToverQ2":
        label = r'$q_T^2/Q^2$'
    if label == "dy":
        label = r'$y_p-y_h$'
    if label == "yh_minus_yp":
        label = r'$y_h-y_p$'
    if label == "yi":
        label = r'$y_i$'
    if label == "yf":
        label = r'$y_f$'
    if label == "yh":
        label = r'\boldmath{$y_h$}'
    if label == "yp":
        label = r'$y_p$'
    if label == "yi_minus_yp":
        label = r'$|y_i - y_p|$'
    if label == "yf_minus_yh":
        label = r'$|y_f - y_h|$'
    if label == "yi_minus_yp_over_yp":
        label = r'$|(y_i - y_p)/yp|$'
    if label == "yf_minus_yh_over_yh":
        label = r'$|(y_f - y_h)/yh|$'
    if label == "R":
        label = r'$|R|$'    
    if label == "lnR":
        label = r'$ln(|R|)$'    
    if label == "R2":
        label = r'$R_2$'    
    if label == "R3":
        label = r'$R_3$'    
    if label == "R4":
        label = r'$R_4$'    
    if label == "R5":
        label = r'$R_5$'    
    if label == "R1":
        label = r'$R_1$'    
    if label == "R1p":
        label = r"$R'_1$"    
    if label == "R0":
        label = r'$R_0$'    
    if label == "x":
        label = r'\boldmath{$x_{\rm Bj}$}'    


    return label


def custom_label1(label):
    
    if label == "qT":
        label = 'q_T\; (GeV)'   
    if label == "pT":
        label = 'P_{hT}\; (GeV)'   


    return label



def color_plot(data, vert_lab, hor_lab, cmap="plasma", alpha=1.0, cbarshow = False):
    vert = data[vert_lab].values
    hor = data[hor_lab].values

    colors = data["tmdaff"]

    
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='minor', labelsize=20, direction='in')


    ax.set_ylabel(custom_label(vert_lab),
                        size=20)
    ax.set_xlabel(custom_label(hor_lab),
                        size=20)

    if cmap == "none": 
        plot = ax.scatter(hor, vert, c="b", alpha=alpha)
    else:
        plot = ax.scatter(hor, vert, c=colors, cmap=plt.get_cmap(cmap), alpha=alpha)
        
    if cbarshow:
        cbar = plt.colorbar(plot, pad=0.05)
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_label(r'$\rm TMD \;affinity$',
                        size=25)

    fig.tight_layout()
    
    return fig, ax, vert, hor


def color_plot_hull(data, vert_lab, hor_lab, cmap="plasma", alpha=1.0, color='b', cbarshow = False):
    vert = data[vert_lab].values
    hor = data[hor_lab].values

    colors = data["tmdaff"]

    
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='minor', labelsize=20, direction='in')


    ax.set_ylabel(custom_label(vert_lab),
                        size=20)
    ax.set_xlabel(custom_label(hor_lab),
                        size=20)

    if cmap == "none": 
        plot = ax.scatter(hor, vert, c=color, alpha=alpha)
    else:
        plot = ax.scatter(hor, vert, c=colors, cmap=plt.get_cmap(cmap), alpha=alpha)
        
    points = np.ndarray(shape=(len(vert),2), dtype=float)
    for i in range(len(vert)):
        points[i,0] = hor[i]
        points[i,1] = vert[i]
        
    hull = ConvexHull(points)
    
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color+'-', alpha=alpha) 
    ax.fill(points[hull.vertices,0], points[hull.vertices,1], color, alpha=0.3)

        
    if cbarshow:
        cbar = plt.colorbar(plot, pad=0.05)
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_label(r'$\rm TMD \;affinity$',
                        size=25)

    fig.tight_layout()
    
    return fig, ax, vert, hor


def above(dict,i,k):
      return (i-1,k) in dict.keys()

def below(dict,i,k):
      return (i+1,k) in dict.keys()

def left(dict,i,k):
      return (i,k-1) in dict.keys()

def right(dict,i,k):
      return (i,k+1) in dict.keys()

def plotJLab12(data, hadron = 'pi+', affinity = 'tmdaff', plotx = 'qT', ploty = 'z', cmap_name = 'seismic_r', yscale = 'linear'):

    if 'Q' not in data.keys():
        data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
        
        
    Q2b=data.Q2.unique()    
    xb=data.x.unique()
    zbins=data.z.unique()    
    
    bins={}
    
    for ix in range(len(xb)):
        for iQ2 in range(len(Q2b)):
            #print "iQ2=", len(Q2b)-iQ2-1, " ix= ", ix, ": ","Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            msg="Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            if data.query(msg).index.size != 0:
                bins[(len(Q2b)-iQ2-1,ix)]=msg

    
    
    nrows,ncols=len(Q2b),len(xb)
    fig = py.figure(figsize=(ncols*3.2,nrows*3.2))
    gs = gridspec.GridSpec(nrows,ncols)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.86,bottom=0.13,top=0.86)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
 
    # add a smaller subplot to explain axes
    leftb, bottomb, widthb, heightb = [0.2, 0.6, 0.25, 0.2]
    ax2 = fig.add_axes([leftb, bottomb, widthb, heightb])
    
    for k in sorted(bins):
        ir,ic=k
        #print k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        ax.set_ylim(0,1)
        #ax.set_xlim(0,data.qT.max())
        if ploty == 'z': 
            ax.set_xlim(0,1) # z is in [0,1]
            ax2.set_xlim(0,1)
            ax2.set_xlabel(r'$z_h$', fontsize=70) 
        if plotx == 'pT': 
            ax.set_ylim(0,3) # pT is in [0,2]
            ax2.set_ylim(0,3)
            ax2.set_ylabel(r'\boldmath{$P_{hT}\;\rm (GeV)$}', fontsize=70) 
        if plotx == 'qT': 
            ax.set_ylim(0,15) #(0,data.qT.max())
            ax2.set_ylim(0,15)
            ax2.set_ylabel(r'$q_T$', fontsize=70)
            
                     
            
        ax.set_yscale(yscale) # log or linear
        ax2.set_yscale(yscale)
        
        # Plot 5 ticks on x and y axis and drop the first and the last ones to avoid overlay:
        xticks = np.round(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],5),1)[1:4]
        yticks = np.round(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],5),1)[1:4]
        
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_xticklabels(xticks, fontsize=40)  
        ax2.set_yticklabels(yticks, fontsize=40)
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if  below(bins,ir,ic)==False : # no bins below
            ax.set_xticklabels(xticks)
        if  left(bins,ir,ic)==False : # no bins to the left
            ax.set_yticklabels(yticks)   
        
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        
        for i in range(len(zbins)):
            #somehow simple query does not work:
            #dd=d.query('z==%f'%zbins[i])
            msg='z > '+str(zbins[i]-zbins[i]/100)+' and z < '+ str(zbins[i]+zbins[i]/100)
            dd=d.query(msg)
            if dd.index.size==0: continue
            #plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity], c=dd[affinity], 
            #                      cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            #ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            plot = ax.scatter(dd[ploty],dd[plotx], s=1500*dd[affinity]+10, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
            #ax.text(0, 2, k, fontsize=18) # show what bin is shown
            if k == (2,3):
                ax2.scatter(dd[ploty],dd[plotx], s=3500*dd[affinity]+10, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
                ax2.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
                ax.annotate('',xy=(0.,1),xycoords='axes fraction',xytext=(-0.75,1.5), 
                            arrowprops=dict(arrowstyle="->, head_width=1, head_length=2", color='k',lw=4))
                  
                
                
        ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
        
        
        # Add embelishment here:
        if  below(bins,ir,ic)==False and left(bins,ir,ic)==False:    

            ax.annotate('', xy=(-0.35, 8.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(5.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'\boldmath{$Q^2~({\rm GeV}^2)$}', 
                        xy=(-1.5,4),
                        xycoords='axes fraction',
                        size=80,
                        rotation=90)

            ax.annotate(r'\boldmath{$x_{\rm Bj}$}', 
                        xy=(2.3,-1.2),
                        xycoords='axes fraction',
                        size=80)
                    
            for i in range(len(data.x.unique())):
                if xb[i]<2e-3: msg=r'$%0.5f$'%xb[i]
                elif xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]  
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.65,msg,transform=ax.transAxes,size=45,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(len(data.Q2.unique())):
                ax.text(-0.65,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=45,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        #if plotx == 'qT': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
            
        
        if below(bins,ir,ic)==False and left(bins,ir,ic)==False:    # otherwise just plot qt>Q
            label1 = ' '
            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'



            #msg=r'${\rm %s~region~EIC~%s}$'%(label1,hadron)
            msg=r'\boldmath{${\rm %s~region~JLab}$}'%(label1)
            ax.text(0,8.2,msg,transform=ax.transAxes,size=80)
            #msg =r'\boldmath{${\sqrt{s}=4.6 \; \; \rm GeV}$}'
            #ax.text(0,8.2,msg,transform=ax.transAxes,size=80)
            #msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            #ax.text(0,5.2,msg,transform=ax.transAxes,size=80)
            
            # plot the legend of axes
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
    

    
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=40)
    outname = 'JLab12_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    #py.savefig('Figs/%s.pdf'%outname)
    py.savefig('./Figs/%s.pdf'%outname,
            bbox_inches ="tight")
    #py.savefig('./Figs/%s.pdf'%outname,
    #        bbox_inches ="tight",
    #        pad_inches = 1)      



def plotHermes(data , hadron = 'pi+', affinity = 'tmdaff', cmap_name = 'seismic_r', yscale = 'log', plotx = 'qT', ploty = 'value', plotdata = False, predictions = False):

    
    if 'Q' not in data.keys():
        data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
    
    bins={}
    bins[(1,0)]="Q2>=1. and 0.0023<=x and x<=0.047 "
    bins[(1,1)]="Q2>=1. and 0.047<x  and x<=0.075 "
    bins[(1,2)]="Q2>=1. and 0.075<x  and x<=0.12 "
    bins[(1,3)]="Q2>=1. and 0.12<x   and x<=0.2 "
    bins[(1,4)]="Q2>=1. and 0.2<x    and x<=0.35 "
    bins[(1,5)]="Q2>=1. and 0.35<x    and x<=0.6 "




    Q2bins={}
    Q2bins[(0,0)]="Q2>=1. and 0.023<=x  and x<=0.6 "
    Q2b=[]
    for k in Q2bins:
        d=data.query('%s and  had=="%s"'%(Q2bins[k],hadron))
        Q2b.append(d.Q2.mean())
    Q2b=np.sort(np.unique(Q2b))

    #http://hermesmults.appspot.com/#data
    #Balanced binning in x, z and Ph⊥
#Name: zxpt-3D
#Profile: x: 6 / z: 8 / Ph⊥: 7
#Use for: Analysis that benefit from a balanced full binning profile.
#Edges:
#Variable	Edges
#Q2 [GeV2]	> 1
#x	0.023 - 0.047 - 0.075 - 0.12 - 0.2 - 0.35 - 0.6
#z	0.1 - 0.2 - 0.25 - 0.3 - 0.375 - 0.475 - 0.6 - 0.8 - 1.1
#Ph⊥ [GeV]	0.0 - 0.15 - 0.25 - 0.35 - 0.45 - 0.6 - 0.8 - 1.2
    #0.023 - 0.047 - 0.075 - 0.12 - 0.2 - 0.35 - 0.6
    xbins={}
    xbins[(1,0)]="0.0023<=x and x<=0.047 "
    xbins[(1,1)]="0.047<x  and x<=0.075 "
    xbins[(1,2)]="0.075<x  and x<=0.12 "
    xbins[(1,3)]="0.12<x   and x<=0.2 "
    xbins[(1,4)]="0.2<x    and x<=0.35 "
    xbins[(1,5)]="0.35<x    and x<=0.6 "
    
    xb=[]
    for k in xbins:
        d=data.query('%s and  had=="%s"'%(xbins[k],hadron))
        xb.append(d.x.mean())
    xb=np.sort(np.unique(xb))


    #zbins=[[0.1,0.15],[0.15,0.2],[0.2,0.25],[0.25,0.3],[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,1.1]]
    #0.1 - 0.2 - 0.25 - 0.3 - 0.375 - 0.475 - 0.6 - 0.8 - 1.1
    zbins=[[0.1,0.2],[0.2,0.25],[0.25,0.3],[0.3,0.375],[0.375,0.475],[0.475,0.6],[0.6,0.8],[0.8,1.1]]

    nrows,ncols=2,7
    fig = py.figure(figsize=(ncols*3.2,nrows*3.2))
    gs = gridspec.GridSpec(2,7)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.8,bottom=0.25,top=0.8)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
  
    
    for k in sorted(bins):
        ir,ic=k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8) # limits for qT
        if ploty == 'value': ax.set_ylim(1e-4,100)
        if ploty == 'z': ax.set_ylim(0,1)
            
        if plotx == 'qT': 
            ticksx = [0,2,4,6]
        elif plotx == 'pT': 
            ticksx = [0,0.2,0.4,0.6,0.8]
            ax.set_xlim(0,1)
            
        ax.set_yscale(yscale) # log or lin
        if all(k!=_ for _ in [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8)]): 
            ax.set_xticks([]) 
        else:
            ax.set_xticks(ticksx)
            
        if all(k!=_ for _ in [(1,0)]): 
            ax.set_yticklabels([])
        else:    
            if ploty == 'value': ax.set_yticks([1e-3,1e-2,1e-1,1,10])            
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        for i in range(len(zbins)):
        
            
            msg='%f<z and z<%f'%(zbins[i][0],zbins[i][1])
            dd=d.query(msg)
            if dd.index.size==0: continue
            plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity]**0.2+20, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            
            if ploty == 'value' and plotdata:
                #e=ax.errorbar(dd.qT,dd.value,np.sqrt(dd.stat_u**2 + dd.systabs_u**2),fmt='.',label=r'$%0.2f<z_h<%0.2f$'%(zbins[i][0],zbins[i][1])) 
                e=ax.errorbar(dd[plotx],dd.value,np.sqrt(dd.stat_u**2 + dd.systabs_u**2),fmt='.',color='k',label=r'$%0.2f<z_h<%0.2f$'%(zbins[i][0],zbins[i][1])) 
                c=e[0].get_color() # Plot the data is value is plotted TODO, what is alpha?

            if predictions and ploty == 'value' and 'thy' in dd.keys():
                # Plot TMD prediction for those bins
                #ax.fill_between(dd[plotx],(dd['Prediction']+dd['Prediction_err']),(dd['Prediction']-dd['Prediction_err']),alpha=0.25) 
                #ax.plot(dd[plotx],dd['Prediction'],'b-', alpha=0.5,label='')  # Plot TMD JAM20 SIDIS 
                ax.plot(dd[plotx],dd['thy'],'g-', alpha=0.5,label='')  # Plot TMD JAM20 SIDIS 

                
            if predictions and ploty == 'value' and 'LO' in dd.keys():
                tot=dd.DDSLO+dd.DDSNLOdelta+dd.DDSNLOplus+dd.DDSNLOrest # DDS NLO from Wang, Rogers, Sato, Gonzalez
                tot=tot.values
                ax.plot(dd.qT,tot/dd.idisNLO*(2.*dd.pT),color='r',ls='--',label='',alpha=0.5) # Plot NLO SIDIS (CONVERT TO HERMES notations!)
    
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
        
        
        if k==(1,0):

            ax.annotate('', xy=(-0.35, 2.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(6.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'\boldmath{$Q^2~({\rm GeV}^2)$}', 
                        xy=(-1,0.5),
                        xycoords='axes fraction',
                        size=40,
                        rotation=90)

            ax.annotate(r'\boldmath{$x_{\rm Bj}$}', 
                        xy=(2.9,-0.8),
                        xycoords='axes fraction',
                        size=40)
                    
            for i in range(6):
                if xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.6,msg,transform=ax.transAxes,size=30,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(1):
                ax.text(-0.6,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=30,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        if ploty == 'value' and plotx=='qT': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
        
        if k==(1,0): 
            if predictions: # if we plot predictions, explain what those are
                TMD=py.Line2D([0,2], [0,0], color='b',ls='-',alpha=0.5)
                NLO=py.Line2D([0], [0], color='r',ls='--',alpha=0.5)
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([(TMD),NLO,qTrange],[r'$\rm TMD$',r'$\rm NLO$',r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[4.8, 1.5]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot
            elif plotx == 'qT': # otherwise just plot qt>Q
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([qTrange],[r'$q_{T}>Q$']\
                    ,bbox_to_anchor=[4.8, 1.5]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot  
                
            #msg=r'${\rm %s~region~HERMES~%s}$'%(affinity,hadron)
            #msg=r'${\rm %s~region~HERMES}$'%(affinity)
            #ax.text(0.1,1.8,msg,transform=ax.transAxes,size=50)
            #msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            #ax.text(0.1,1.2,msg,transform=ax.transAxes,size=50)
            label1 = ' '
            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            #msg=r'${\rm %s~region~EIC~%s}$'%(label1,hadron)
            msg=r'\boldmath{${\rm %s~region~HERMES}$}'%(label1)
            ax.text(0.1,1.8,msg,transform=ax.transAxes,size=40)
            if ploty=='value':
                msg =r'\boldmath{${\rm M^h_n~vs.~%s}$}'%(custom_label1(plotx))
            else:    
                msg =r'${\rm %s~vs.~%s}$'%(ploty,custom_label1(plotx))

            ax.text(0.1,1.2,msg,transform=ax.transAxes,size=40)


            
            
        if k==(1,4): # plot the legend of z binning
            ax.legend(bbox_to_anchor=[2.8, 0.78], loc='center',fontsize=25,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
        
    
    cbar_ax = fig.add_axes([0.86, 0.2, 0.01, 0.5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    outname = 'HERMES_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('Figs/%s.pdf'%outname)    

import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

def smooth(X,Y):
    x=[X[j] for j in range(X.size) if np.isnan(Y[j])==False]
    y=[Y[j] for j in range(Y.size) if np.isnan(Y[j])==False]
    f = interp1d(x, y,fill_value="extrapolate")
    for i in range(1,X.size-1):
        if np.isnan(Y[i]): Y[i]=f(X[i])

    for i in range(X.size):
        x=[X[j] for j in range(X.size) if j!=i]
        y=[Y[j] for j in range(Y.size) if j!=i]
        f = interp1d(x, y,fill_value="extrapolate")
        rel_err=abs(f(X[i])-Y[i])/f(X[i])
        if rel_err>0.05: Y[i]=f(X[i])
    return Y

def plotCompass(data, hadron = 'h+', affinity = 'tmdaff', cmap_name = 'seismic_r', yscale = 'log', plotx = 'qT', ploty = 'value', plotdata = False, predictions = False):

    
    if 'Q' not in data.keys():
        data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']   
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
    
    bins={}
    bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    bins[(0,7)]="17<Q2 and 0.2<x "
    bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    bins[(1,5)]="6<Q2 and Q2<17 and 0.06<x  and x<0.09 "
    bins[(1,6)]="6<Q2 and Q2<17 and 0.09<x  and x<0.2 "
    bins[(1,7)]="6<Q2 and Q2<17 and 0.2<x "
    bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    bins[(2,3)]="3<Q2 and Q2<6 and 0.02<x  and x<0.03 "
    bins[(2,4)]="3<Q2 and Q2<6 and 0.03<x  and x<0.06 "
    bins[(2,5)]="3<Q2 and Q2<6 and 0.06<x  and x<0.09 "
    bins[(2,6)]="3<Q2 and Q2<6 and 0.09<x  and x<0.2 "
    bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    bins[(3,1)]="1.5<Q2 and Q2<3.0 and 0.009<x and x<0.012 "
    bins[(3,2)]="1.5<Q2 and Q2<3.0 and 0.012<x and x<0.02 "
    bins[(3,3)]="1.5<Q2 and Q2<3.0 and 0.02<x  and x<0.03 "
    bins[(3,4)]="1.5<Q2 and Q2<3.0 and 0.03<x  and x<0.06 "
    bins[(3,5)]="1.5<Q2 and Q2<3.0 and 0.06<x  and x<0.09 "
    bins[(4,0)]="Q2<1.5 and x<0.009 "
    bins[(4,1)]="Q2<1.5 and 0.009<x and x<0.012 "
    bins[(4,2)]="Q2<1.5 and 0.012<x and x<0.02 "
    bins[(4,3)]="Q2<1.5 and 0.02<x  and x<0.03 "
    bins[(4,4)]="Q2<1.5 and 0.03<x  and x<0.06 "

    Q2bins={}
    Q2bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    Q2bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    Q2bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    Q2bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    Q2bins[(4,0)]="Q2<1.5 and x<0.009 "
    Q2b=[]
    for k in Q2bins:
        d=data.query('%s and  had=="%s"'%(Q2bins[k],hadron))
        Q2b.append(d.Q2.mean())
    Q2b=np.sort(np.unique(Q2b))

    xbins={}
    xbins[(4,0)]="x<0.009 "
    xbins[(4,1)]="0.009<x and x<0.012 "
    xbins[(4,2)]="0.012<x and x<0.02 "
    xbins[(4,3)]="0.02<x  and x<0.03 "
    xbins[(4,4)]="0.03<x  and x<0.06 "
    xbins[(2,3)]="0.02<x  and x<0.03 "
    xbins[(2,4)]="0.03<x  and x<0.06 "
    xbins[(2,5)]="0.06<x  and x<0.09 "
    xbins[(2,6)]="0.09<x  and x<0.2 "
    xbins[(1,7)]="0.2<x "
    
    xb=[]
    for k in xbins:
        d=data.query('%s and  had=="%s"'%(xbins[k],hadron))
        xb.append(d.x.mean())
    xb=np.sort(np.unique(xb))


    zbins=[[0.24,0.3],[0.3,0.4],[0.4,0.5],[0.65,0.7]]

    nrows,ncols=5,8
    fig = py.figure(figsize=(ncols*3,nrows*3.1))
    gs = gridspec.GridSpec(5,8)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.96,bottom=0.13,top=0.96)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
  
    
    for k in sorted(bins):
        ir,ic=k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        if ploty == 'value': ax.set_ylim(1e-4,10)
        if ploty == 'z': ax.set_ylim(0,1)
            
        ax.set_yscale(yscale) # log or lin
        if all(k!=_ for _ in [(4,0),(4,1),(4,2),(4,3),(4,4),(3,5),(2,6),(1,7)]): 
            ax.set_xticklabels([])
        else:
            ax.set_xticks([0,2,4,6])
        if k==(1,7) or k==(2,6) or k==(3,5):
            ax.set_xticks([2,4,6,8])        
        if all(k!=_ for _ in [(4,0),(3,0),(2,2),(1,4),(0,6)]): 
            ax.set_yticklabels([])
        else:    
            if ploty == 'value': ax.set_yticks([1e-3,1e-2,1e-1,1])            
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        for i in range(len(zbins)):
        
            
            msg='%f<z and z<%f'%(zbins[i][0],zbins[i][1])
            dd=d.query(msg)
            if dd.index.size==0: continue
            plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity]**0.2+20, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            
            if ploty == 'value' and plotdata:
                #e=ax.errorbar(dd.qT,dd.value,dd.alpha,fmt='.',label=r'$%0.2f<z<%0.2f$'%(zbins[i][0],zbins[i][1]))
                #e=ax.errorbar(dd.qT,dd.value,np.sqrt(dd.stat_u**2+dd.syst_u**2),fmt='.',label=r'$%0.2f<z_h<%0.2f$'%(zbins[i][0],zbins[i][1]))
                e=ax.errorbar(dd.qT,dd.value,np.sqrt(dd.stat_u**2+dd.syst_u**2),fmt='.',color='k',label=r'$%0.2f<z_h<%0.2f$'%(zbins[i][0],zbins[i][1]))
                c=e[0].get_color() # Plot the data is value is plotted TODO, what is alpha?

            if predictions and ploty == 'value' and 'Prediction' in dd.keys():
                # Plot TMD prediction for those bins
                ax.fill_between(dd[plotx],dd['Prediction']+dd['Prediction_err'],dd['Prediction']-dd['Prediction_err'],alpha=0.25)
                ax.plot(dd[plotx],dd['Prediction'],'b-', alpha=0.5,label='')  # Plot TMD JAM20 SIDIS
                
            if predictions and ploty == 'value' and 'LO' in dd.keys():
                tot=dd.LO+dd.NLOdelta+dd.NLOplus+dd.NLOregular # NLO from Wang, Rogers, Sato, Gonzalez
                tot=tot.values
                tot=smooth(dd.qT.values,tot)
                ax.plot(dd.qT,tot/dd.idisNLO,color='r',ls='--',label='',alpha=0.5) # Plot NLO SIDIS
    
        ax.tick_params(axis='both', which='major', labelsize=30, direction='in')
        
        
        if k==(4,0):

            ax.annotate('', xy=(-0.35, 5.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(8.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'\boldmath{$Q^2~({\rm GeV}^2)$}', 
                        xy=(-1,2),
                        xycoords='axes fraction',
                        size=50,
                        rotation=90)

            ax.annotate(r'\boldmath{$x_{\rm Bj}$}', 
                        xy=(3.9,-0.7),
                        xycoords='axes fraction',
                        size=50)
                    
            for i in range(8):
                if xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.5,msg,transform=ax.transAxes,size=30,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(5):
                ax.text(-0.6,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=30,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        if ploty == 'value': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
        if k==(2,2):
            if predictions: # if we plot predictions, explain what those are
                TMD=py.Line2D([0,2], [0,0], color='b',ls='-',alpha=0.5)
                NLO=py.Line2D([0], [0], color='r',ls='--',alpha=0.5)
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([(TMD),NLO,qTrange],[r'$\rm TMD$',r'$\rm NLO$',r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot
            else: # otherwise just plot qt>Q
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([qTrange],[r'\boldmath{$q_{\rm T}>Q$}']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot  

            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

                
                
                
            msg=r'\boldmath{${\rm %s~region~COMPASS}$}'%(label1)
            ax.text(-2,2.8,msg,transform=ax.transAxes,size=50)
            if ploty=='value':
                msg =r'\boldmath{${\rm M^h~vs.~%s}$}'%(custom_label1(plotx))
            else:    
                msg =r'${\rm %s~vs.~%s}$'%(ploty,custom_label1(plotx))
            ax.text(-2,2.2,msg,transform=ax.transAxes,size=50)
            
        if k==(1,4): # plot the legend of
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
        
    
    cbar_ax = fig.add_axes([.95, 0.1, 0.01, .5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    outname = 'COMPASS_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('Figs/%s.pdf'%outname)    


def plotCompass1(data, hadron = 'h+', affinity = 'tmdaff', cmap_name = 'seismic_r', yscale = 'log', plotx = 'qT', ploty = 'value', plotdata = False, predictions = False):

    
    data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']
    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
    
    bins={}
    bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    bins[(0,7)]="17<Q2 and 0.2<x "
    bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    bins[(1,5)]="6<Q2 and Q2<17 and 0.06<x  and x<0.09 "
    bins[(1,6)]="6<Q2 and Q2<17 and 0.09<x  and x<0.2 "
    bins[(1,7)]="6<Q2 and Q2<17 and 0.2<x "
    bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    bins[(2,3)]="3<Q2 and Q2<6 and 0.02<x  and x<0.03 "
    bins[(2,4)]="3<Q2 and Q2<6 and 0.03<x  and x<0.06 "
    bins[(2,5)]="3<Q2 and Q2<6 and 0.06<x  and x<0.09 "
    bins[(2,6)]="3<Q2 and Q2<6 and 0.09<x  and x<0.2 "
    bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    bins[(3,1)]="1.5<Q2 and Q2<3.0 and 0.009<x and x<0.012 "
    bins[(3,2)]="1.5<Q2 and Q2<3.0 and 0.012<x and x<0.02 "
    bins[(3,3)]="1.5<Q2 and Q2<3.0 and 0.02<x  and x<0.03 "
    bins[(3,4)]="1.5<Q2 and Q2<3.0 and 0.03<x  and x<0.06 "
    bins[(3,5)]="1.5<Q2 and Q2<3.0 and 0.06<x  and x<0.09 "
    bins[(4,0)]="Q2<1.5 and x<0.009 "
    bins[(4,1)]="Q2<1.5 and 0.009<x and x<0.012 "
    bins[(4,2)]="Q2<1.5 and 0.012<x and x<0.02 "
    bins[(4,3)]="Q2<1.5 and 0.02<x  and x<0.03 "
    bins[(4,4)]="Q2<1.5 and 0.03<x  and x<0.06 "

    Q2bins={}
    Q2bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    Q2bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    Q2bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    Q2bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    Q2bins[(4,0)]="Q2<1.5 and x<0.009 "
    Q2b=[]
    for k in Q2bins:
        d=data.query('%s and  had=="%s"'%(Q2bins[k],hadron))
        Q2b.append(d.Q2.mean())
    Q2b=np.sort(np.unique(Q2b))

    xbins={}
    xbins[(4,0)]="x<0.009 "
    xbins[(4,1)]="0.009<x and x<0.012 "
    xbins[(4,2)]="0.012<x and x<0.02 "
    xbins[(4,3)]="0.02<x  and x<0.03 "
    xbins[(4,4)]="0.03<x  and x<0.06 "
    xbins[(2,3)]="0.02<x  and x<0.03 "
    xbins[(2,4)]="0.03<x  and x<0.06 "
    xbins[(2,5)]="0.06<x  and x<0.09 "
    xbins[(2,6)]="0.09<x  and x<0.2 "
    xbins[(1,7)]="0.2<x "
    
    xb=[]
    for k in xbins:
        d=data.query('%s and  had=="%s"'%(xbins[k],hadron))
        xb.append(d.x.mean())
    xb=np.sort(np.unique(xb))


    zbins=[[0.24,0.3],[0.3,0.4],[0.4,0.5],[0.65,0.7]]

    nrows,ncols=5,8
    fig = py.figure(figsize=(ncols*3,nrows*3.1))
    gs = gridspec.GridSpec(5,8)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.96,bottom=0.13,top=0.96)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
  
    
    for k in sorted(bins):
        ir,ic=k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        if ploty == 'value': ax.set_ylim(1e-4,10)
        if ploty == 'z': ax.set_ylim(0,1)
            
        ax.set_yscale(yscale) # log or lin
        if all(k!=_ for _ in [(4,0),(4,1),(4,2),(4,3),(4,4),(3,5),(2,6),(1,7)]): 
            ax.set_xticklabels([])
        else:
            ax.set_xticks([0,2,4,6])
        if k==(1,7) or k==(2,6) or k==(3,5):
            ax.set_xticks([2,4,6,8])        
        if all(k!=_ for _ in [(4,0),(3,0),(2,2),(1,4),(0,6)]): 
            ax.set_yticklabels([])
        else:    
            if ploty == 'value': ax.set_yticks([1e-3,1e-2,1e-1,1])            
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        for i in range(len(zbins)):
        
            
            msg='%f<z and z<%f'%(zbins[i][0],zbins[i][1])
            dd=d.query(msg)
            if dd.index.size==0: continue
            plot = ax.scatter(dd[plotx],dd[ploty], s=200*dd[affinity], c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            
            if ploty == 'value' and plotdata:
                e=ax.errorbar(dd.qT,dd.value,dd.alpha,fmt='.',label=r'$%0.2f<z<%0.2f$'%(zbins[i][0],zbins[i][1]))
                c=e[0].get_color() # Plot the data is value is plotted TODO, what is alpha?

            if predictions and ploty == 'value' and 'Prediction' in dd.keys():
                # Plot TMD prediction for those bins
                ax.fill_between(dd[plotx],dd['Prediction']+dd['Prediction_err'],dd['Prediction']-dd['Prediction_err'],alpha=0.25)
                ax.plot(dd[plotx],dd['Prediction'],'b-', alpha=0.5,label='')  # Plot TMD JAM20 SIDIS
                
            if predictions and ploty == 'value' and 'LO' in dd.keys():
                tot=dd.LO+dd.NLOdelta+dd.NLOplus+dd.NLOregular # NLO from Wang, Rogers, Sato, Gonzalez
                tot=tot.values
                tot=smooth(dd.qT.values,tot)
                ax.plot(dd.qT,tot/dd.idisNLO,color='r',ls='--',label='',alpha=0.5) # Plot NLO SIDIS
    
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
        
        
        if k==(4,0):

            ax.annotate('', xy=(-0.35, 5.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(8.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'$Q^2~({\rm GeV}^2)$', 
                        xy=(-1,3),
                        xycoords='axes fraction',
                        size=50,
                        rotation=90)

            ax.annotate(r'$x_{\rm Bj}$', 
                        xy=(3.9,-0.7),
                        xycoords='axes fraction',
                        size=50)
                    
            for i in range(8):
                if xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.5,msg,transform=ax.transAxes,size=30,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(5):
                ax.text(-0.6,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=30,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        if ploty == 'value': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
        if k==(2,2):
            if predictions: # if we plot predictions, explain what those are
                TMD=py.Line2D([0,2], [0,0], color='b',ls='-',alpha=0.5)
                NLO=py.Line2D([0], [0], color='r',ls='--',alpha=0.5)
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([(TMD),NLO,qTrange],[r'$\rm TMD$',r'$\rm NLO$',r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot
            else: # otherwise just plot qt>Q
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([qTrange],[r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot  

            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

                
                
                
            msg=r'${\rm %s~region~COMPASS}$'%(label1)
            ax.text(-2,2.8,msg,transform=ax.transAxes,size=50)
            if ploty=='value':
                msg =r'${\rm M^h~vs.~%s}$'%(plotx)
            else:    
                msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            ax.text(-2,2.2,msg,transform=ax.transAxes,size=50)
            
        if k==(1,4): # plot the legend of
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
        
    
    cbar_ax = fig.add_axes([.95, 0.1, 0.01, .5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    outname = 'COMPASS_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('Figs/%s.pdf'%outname)


def plotpolarCompass(data, hadron = 'h+', affinity = 'tmdaff', cmap_name = 'seismic_r', yscale = 'linear', plotx = 'qT', ploty = 'value', plotdata = False, predictions = False):

    if 'Q' not in data.keys():
        data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']
    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
    
    bins={}
    bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    bins[(0,7)]="17<Q2 and 0.2<x "
    bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    bins[(1,5)]="6<Q2 and Q2<17 and 0.06<x  and x<0.09 "
    bins[(1,6)]="6<Q2 and Q2<17 and 0.09<x  and x<0.2 "
    bins[(1,7)]="6<Q2 and Q2<17 and 0.2<x "
    bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    bins[(2,3)]="3<Q2 and Q2<6 and 0.02<x  and x<0.03 "
    bins[(2,4)]="3<Q2 and Q2<6 and 0.03<x  and x<0.06 "
    bins[(2,5)]="3<Q2 and Q2<6 and 0.06<x  and x<0.09 "
    bins[(2,6)]="3<Q2 and Q2<6 and 0.09<x  and x<0.2 "
    bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    bins[(3,1)]="1.5<Q2 and Q2<3.0 and 0.009<x and x<0.012 "
    bins[(3,2)]="1.5<Q2 and Q2<3.0 and 0.012<x and x<0.02 "
    bins[(3,3)]="1.5<Q2 and Q2<3.0 and 0.02<x  and x<0.03 "
    bins[(3,4)]="1.5<Q2 and Q2<3.0 and 0.03<x  and x<0.06 "
    bins[(3,5)]="1.5<Q2 and Q2<3.0 and 0.06<x  and x<0.09 "
    bins[(4,0)]="Q2<1.5 and x<0.009 "
    bins[(4,1)]="Q2<1.5 and 0.009<x and x<0.012 "
    bins[(4,2)]="Q2<1.5 and 0.012<x and x<0.02 "
    bins[(4,3)]="Q2<1.5 and 0.02<x  and x<0.03 "
    bins[(4,4)]="Q2<1.5 and 0.03<x  and x<0.06 "

    Q2bins={}
    Q2bins[(0,6)]="17<Q2 and 0.09<x  and x<0.2 "
    Q2bins[(1,4)]="6<Q2 and Q2<17 and 0.03<x  and x<0.06 "
    Q2bins[(2,2)]="3<Q2 and Q2<6 and 0.012<x and x<0.02 "
    Q2bins[(3,0)]="1.5<Q2 and Q2<3.0 and x<0.009 "
    Q2bins[(4,0)]="Q2<1.5 and x<0.009 "
    Q2b=[]
    for k in Q2bins:
        d=data.query('%s and  had=="%s"'%(Q2bins[k],hadron))
        Q2b.append(d.Q2.mean())
    Q2b=np.sort(np.unique(Q2b))

    xbins={}
    xbins[(4,0)]="x<0.009 "
    xbins[(4,1)]="0.009<x and x<0.012 "
    xbins[(4,2)]="0.012<x and x<0.02 "
    xbins[(4,3)]="0.02<x  and x<0.03 "
    xbins[(4,4)]="0.03<x  and x<0.06 "
    xbins[(2,3)]="0.02<x  and x<0.03 "
    xbins[(2,4)]="0.03<x  and x<0.06 "
    xbins[(2,5)]="0.06<x  and x<0.09 "
    xbins[(2,6)]="0.09<x  and x<0.2 "
    xbins[(1,7)]="0.2<x "
    
    xb=[]
    for k in xbins:
        d=data.query('%s and  had=="%s"'%(xbins[k],hadron))
        xb.append(d.x.mean())
    xb=np.sort(np.unique(xb))


    zbins=[[0.24,0.3],[0.3,0.4],[0.4,0.5],[0.65,0.7]]

    nrows,ncols=5,8
    fig = py.figure(figsize=(ncols*3,nrows*3.1))
    gs = gridspec.GridSpec(5,8)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.96,bottom=0.13,top=0.96)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
  
    
    for k in sorted(bins):
        ir,ic=k
        ax = py.subplot(gs[ir,ic], polar = True)
        ax.set_xlim(0,8)
        
        ax.set_thetamin(0)
        ax.set_thetamax(360)
        
        if ploty == 'value': ax.set_ylim(1e-4,10)
        if ploty == 'z': ax.set_ylim(0,1)
            
        ax.set_yscale(yscale) # log or lin
        if all(k!=_ for _ in [(4,0),(4,1),(4,2),(4,3),(4,4),(3,5),(2,6),(1,7)]): 
            ax.set_xticklabels([])
        else:
            ax.set_xticks([0,2,4,6])
        if k==(1,7) or k==(2,6) or k==(3,5):
            ax.set_xticks([2,4,6,8])        
        if all(k!=_ for _ in [(4,0),(3,0),(2,2),(1,4),(0,6)]): 
            ax.set_yticklabels([])
        else:    
            if ploty == 'value': ax.set_yticks([1e-3,1e-2,1e-1,1])            
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        for i in range(len(zbins)):
        
            
            msg='%f<z and z<%f'%(zbins[i][0],zbins[i][1])
            dd=d.query(msg)
            if dd.index.size==0: continue
            plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity], c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            
            if ploty == 'value' and plotdata:
                e=ax.errorbar(dd.qT,dd.value,dd.alpha,fmt='.',label=r'$%0.2f<z<%0.2f$'%(zbins[i][0],zbins[i][1]))
                c=e[0].get_color() # Plot the data is value is plotted TODO, what is alpha?

            if predictions and ploty == 'value' and 'Prediction' in dd.keys():
                # Plot TMD prediction for those bins
                ax.fill_between(dd[plotx],dd['Prediction']+dd['Prediction_err'],dd['Prediction']-dd['Prediction_err'],alpha=0.25)
                ax.plot(dd[plotx],dd['Prediction'],'b-', alpha=0.5,label='')  # Plot TMD JAM20 SIDIS
                
            if predictions and ploty == 'value' and 'LO' in dd.keys():
                tot=dd.LO+dd.NLOdelta+dd.NLOplus+dd.NLOregular # NLO from Wang, Rogers, Sato, Gonzalez
                tot=tot.values
                tot=smooth(dd.qT.values,tot)
                ax.plot(dd.qT,tot/dd.idisNLO,color='r',ls='--',label='',alpha=0.5) # Plot NLO SIDIS
    
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
        
        
        if k==(4,0):

            ax.annotate('', xy=(-0.35, 5.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(8.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'$Q^2~({\rm GeV}^2)$', 
                        xy=(-1,3),
                        xycoords='axes fraction',
                        size=50,
                        rotation=90)

            ax.annotate(r'$x_{\rm Bj}$', 
                        xy=(3.9,-0.7),
                        xycoords='axes fraction',
                        size=50)
                    
            for i in range(8):
                if xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.5,msg,transform=ax.transAxes,size=30,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(5):
                ax.text(-0.6,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=30,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        if ploty == 'value': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
        if k==(2,2):
            if predictions: # if we plot predictions, explain what those are
                TMD=py.Line2D([0,2], [0,0], color='b',ls='-',alpha=0.5)
                NLO=py.Line2D([0], [0], color='r',ls='--',alpha=0.5)
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([(TMD),NLO,qTrange],[r'$\rm TMD$',r'$\rm NLO$',r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot
            else: # otherwise just plot qt>Q
                qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
                ax.legend([qTrange],[r'$q_{\rm T}>Q$']\
                    ,bbox_to_anchor=[-1.2, 1.]\
                    ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot 
                
            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

              
            msg=r'${\rm %s~region~COMPASS~%s}$'%(label1,hadron)
            ax.text(-2,2.8,msg,transform=ax.transAxes,size=50)
            msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            ax.text(-2,2.2,msg,transform=ax.transAxes,size=50)
            
        if k==(1,4): # plot the legend of
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
        
    
    cbar_ax = fig.add_axes([.95, 0.1, 0.01, .5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    outname = 'COMPASS_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('Figs/%s.pdf'%outname)    


def plotEIC(fname, hadron = 'pi+', affinity = 'tmdaff', plotx = 'qT', ploty = 'z', cmap_name = 'seismic_r', yscale = 'linear'):

    data=pd.read_excel(fname)
    data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']
    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
        
        
    Q2b=data.Q2.unique()    
    xb=data.x.unique()
    zbins=data.z.unique()    
    
    bins={}
    
    for ix in range(len(xb)):
        for iQ2 in range(len(Q2b)):
            #print "iQ2=", len(Q2b)-iQ2-1, " ix= ", ix, ": ","Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            msg="Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            if data.query(msg).index.size != 0:
                bins[(len(Q2b)-iQ2-1,ix)]=msg

    
    
    nrows,ncols=len(Q2b),len(xb)
    fig = py.figure(figsize=(ncols*3.2,nrows*3.2))
    gs = gridspec.GridSpec(nrows,ncols)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.86,bottom=0.13,top=0.86)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
  
    
    for k in sorted(bins):
        ir,ic=k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        ax.set_ylim(0,1)
        #ax.set_xlim(0,data.qT.max())
        if ploty == 'z': ax.set_ylim(0,1) # z is in [0,1]
        if plotx == 'pT': ax.set_xlim(0,8) # pT is in [0,2]
        if plotx == 'qT': ax.set_xlim(0,15) #(0,data.qT.max())

            
        ax.set_yscale(yscale) # log or linear
        
        # Plot 5 ticks on x and y axis and drop the first and the last ones to avoid overlay:
        xticks = np.round(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],5),1)[1:4]
        yticks = np.round(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],5),1)[1:4]
            
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if  below(bins,ir,ic)==False : # no bins below
            ax.set_xticklabels(xticks)
        if  left(bins,ir,ic)==False : # no bins to the left
            ax.set_yticklabels(yticks)   
        
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        
        for i in range(len(zbins)):
            #somehow simple query does not work:
            #dd=d.query('z==%f'%zbins[i])
            msg='z > '+str(zbins[i]-zbins[i]/100)+' and z < '+ str(zbins[i]+zbins[i]/100)
            dd=d.query(msg)
            if dd.index.size==0: continue
            plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity], c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
                
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in')
        
        
        # Add embelishment here:
        if  below(bins,ir,ic)==False and left(bins,ir,ic)==False:    

            ax.annotate('', xy=(-0.35, 9.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(17.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'\boldmath{$Q^2~({\rm GeV}^2)$}', 
                        xy=(-1.5,5),
                        xycoords='axes fraction',
                        size=80,
                        rotation=90)

            ax.annotate(r'\boldmath{$x_{\rm Bj}$}', 
                        xy=(7.9,-1.),
                        xycoords='axes fraction',
                        size=90)
                    
            for i in range(len(data.x.unique())):
                if xb[i]<2e-3: msg=r'$%0.5f$'%xb[i]
                elif xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]  
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.65,msg,transform=ax.transAxes,size=45,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(len(data.Q2.unique())):
                ax.text(-0.65,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=45,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        if plotx == 'qT': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
            
        
        if below(bins,ir,ic)==False and left(bins,ir,ic)==False:    # otherwise just plot qt>Q
            #qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
            #ax.legend([qTrange],[r'$q_{\rm T}>Q$']\
            #        ,bbox_to_anchor=[-1.2, 1.]\
            #        ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot  
            label1 = ' '
            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

            #msg=r'${\rm %s~region~EIC~%s}$'%(label1,hadron)
            msg=r'\boldmath{${\rm %s~region~EIC}$}'%(label1)
            ax.text(0,6.8,msg,transform=ax.transAxes,size=80)
            msg =r'\boldmath{${\rm %s~vs.~%s}$}'%(ploty,plotx)
            ax.text(0,5.2,msg,transform=ax.transAxes,size=80)
            
            # plot the legend of axes
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
        
    
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=40)
    outname = 'EIC_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('gallery/%s.pdf'%outname)    


def plotEIC1(data, predictions, hadron = 'pi+', affinity = 'tmdaff', plotx = 'qT', ploty = 'z', cmap_name = 'seismic_r', yscale = 'linear'):

    
    data['Q']=data['Q2']**0.5
    if 'had' not in data.keys() and 'hadron' in data.keys(): 
        data['had'] = data['hadron']
    
    if 'qT' not in data.keys():
        data['qT'] = data['pT']/data['z']
        
    # set affinity < 0 to 0 and affinity > 1 to 1
    predictions[predictions<0] = 0
    predictions[predictions>1] = 1
    data[affinity] = predictions  


        
    Q2b=data.Q2.unique()    
    xb=data.x.unique()
    zbins=data.z.unique()    
    
    bins={}
    
    for ix in range(len(xb)):
        for iQ2 in range(len(Q2b)):
            #print "iQ2=", len(Q2b)-iQ2-1, " ix= ", ix, ": ","Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            msg="Q2=="+str(Q2b[iQ2])+" and x=="+str(xb[ix])
            if data.query(msg).index.size != 0:
                bins[(len(Q2b)-iQ2-1,ix)]=msg

    
    
    nrows,ncols=len(Q2b),len(xb)
    fig = py.figure(figsize=(ncols*3.2,nrows*3.2))
    gs = gridspec.GridSpec(nrows,ncols)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.86,bottom=0.13,top=0.86)
    AX={}
    cmap = plt.get_cmap(cmap_name) # choose cmap
 
    # add a maller subplot to explain axes
    leftb, bottomb, widthb, heightb = [0.3, 0.6, 0.15, 0.2]
    ax2 = fig.add_axes([leftb, bottomb, widthb, heightb])
    
    for k in sorted(bins):
        ir,ic=k
        #print k
        ax = py.subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        ax.set_ylim(0,1)
        #ax.set_xlim(0,data.qT.max())
        if ploty == 'z': 
            ax.set_xlim(0,1) # z is in [0,1]
            ax2.set_xlim(0,1)
            ax2.set_xlabel(r'\boldmath{$z_h$}', fontsize=70) 
        if plotx == 'pT': 
            if affinity.startswith('col'):
                max = 40
            elif affinity.startswith('tmd'):
                max = 10
            elif affinity.startswith('soft'):
                max = 10
            elif affinity.startswith('target'):
                max = 10
            else:
                max = 20
            ax.set_ylim(0,max) # pT is in [0,2]
            ax2.set_ylim(0,max)
            ax2.set_ylabel(r'\boldmath{$P_{hT} \; \rm (GeV)$}', fontsize=70) 
        if plotx == 'qT': 
            ax.set_ylim(0,15) #(0,data.qT.max())
            ax2.set_ylim(0,15)
            ax2.set_ylabel(r'$q_T \; \rm (GeV)$', fontsize=70)
            
 
                    
            
        ax.set_yscale(yscale) # log or linear
        ax2.set_yscale(yscale)
        
        # Plot 5 ticks on x and y axis and drop the first and the last ones to avoid overlay:
        xticks = np.round(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],5),1)[1:4]
        yticks = np.round(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],5),1)[1:4]
        
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_xticklabels(xticks, fontsize=55)  
        ax2.set_yticklabels(yticks, fontsize=55)
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if  below(bins,ir,ic)==False : # no bins below
            ax.set_xticklabels(xticks)
        if  left(bins,ir,ic)==False : # no bins to the left
            ax.set_yticklabels(yticks)   
        
        
        d=data.query('%s and  had=="%s"'%(bins[k],hadron))
        
        for i in range(len(zbins)):
            #somehow simple query does not work:
            #dd=d.query('z==%f'%zbins[i])
            msg='z > '+str(zbins[i]-zbins[i]/100)+' and z < '+ str(zbins[i]+zbins[i]/100)
            dd=d.query(msg)
            if dd.index.size==0: continue
            #plot = ax.scatter(dd[plotx],dd[ploty], s=500*dd[affinity], c=dd[affinity], 
            #                      cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            #ax.plot(dd[plotx],dd[ploty],'k-', alpha=0.25,label='') 
            plot = ax.scatter(dd[ploty],dd[plotx], s=500*dd[affinity]**0.2+10, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
            #ax.text(0, 2, k, fontsize=18) # show what bin is shown
            if k == (3,9):
                ax2.scatter(dd[ploty],dd[plotx], s=2500*dd[affinity]**0.2+20, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
                ax2.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
                ax.annotate('',xy=(0.,1),xycoords='axes fraction',xytext=(-1.8,2), 
                            arrowprops=dict(arrowstyle="->, head_width=1, head_length=2", color='k',lw=4))
                  
                
                
        ax.tick_params(axis='both', which='major', labelsize=32, direction='in')
        
        
        # Add embelishment here:
        if  below(bins,ir,ic)==False and left(bins,ir,ic)==False:    

            ax.annotate('', xy=(-0.35, 9.2), 
                        xycoords='axes fraction', 
                        xytext=(-0.35, -0.1),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))

            ax.annotate('', xy=(16.2,-0.3), 
                        xycoords='axes fraction', 
                        xytext=(-0.1, -0.3),
                        arrowprops=dict(arrowstyle="-|>, head_width=1, head_length=2", 
                        color='k',lw=3))        

            ax.annotate(r'\boldmath{$Q^2~({\rm GeV}^2)$}', 
                        xy=(-1.5,3.5),
                        xycoords='axes fraction',
                        size=80,
                        rotation=90)

            ax.annotate(r'\boldmath{$x_{\rm Bj}$}', 
                        xy=(7.9,-1.2),
                        xycoords='axes fraction',
                        size=90)
                    
            for i in range(len(data.x.unique())):
                if xb[i]<2e-3: msg=r'$%0.5f$'%xb[i]
                elif xb[i]<2e-2: msg=r'$%0.3f$'%xb[i]  
                else:msg=r'$%0.2f$'%xb[i]
                ax.text(0.5+i,-0.65,msg,transform=ax.transAxes,size=55,ha="center")
                ax.annotate('',xy=(i,-0.35),xycoords='axes fraction',xytext=(i+1, -0.35), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
    
            for i in range(len(data.Q2.unique())):
                ax.text(-0.65,0.5+i,r'$%0.1f$'%Q2b[i],
                      transform=ax.transAxes,size=55,rotation=90,va="center")
                ax.annotate('',xy=(-0.4,i),xycoords='axes fraction',xytext=(-0.4,i+1), 
                            arrowprops=dict(arrowstyle="<->", color='k'))
                
        #if plotx == 'qT': ax.plot([d.Q.values[0],8],[2e-4,2e-4],c='y',lw=10,alpha=0.5) # plot qt>Q region
            
        
        if below(bins,ir,ic)==False and left(bins,ir,ic)==False:    # otherwise just plot qt>Q
            #qTrange = mpatches.Rectangle((0,0), 0, 0, ec="none",color='y',alpha=0.5)
            #ax.legend([qTrange],[r'$q_{\rm T}>Q$']\
            #        ,bbox_to_anchor=[-1.2, 1.]\
            #        ,loc='center',fontsize=40,frameon=0) # legend for the lines in the plot  
            label1 = ' '
            if affinity.startswith('tmd'): 
                label1 = 'TMD'
            elif affinity.startswith('col'): 
                label1 = 'Collinear' 
            elif affinity.startswith('target'): 
                label1 = 'Target' 
            elif affinity.startswith('soft'): 
                label1 = 'Soft'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

            #msg=r'${\rm %s~region~EIC~%s}$'%(label1,hadron)
            msg=r'\boldmath{${\rm %s~region~EIC}$}'%(label1)
            ax.text(0,8.8,msg,transform=ax.transAxes,size=80)
            #msg =r'${\sqrt{s}=140 \; \; \rm GeV}$'
            #ax.text(0,8.2,msg,transform=ax.transAxes,size=80)
            #msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            #ax.text(0,5.2,msg,transform=ax.transAxes,size=80)
            
            # plot the legend of axes
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
    

    
    cbar_ax = fig.add_axes([0.87, 0.2, 0.01, 0.5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=40)
    outname = 'EIC_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    py.savefig('Figs/%s.pdf'%outname)    