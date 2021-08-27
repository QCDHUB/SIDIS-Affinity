import numpy as np
from numpy import round, linspace
import pandas as pd
from functools import partial, reduce
from scipy.stats import percentileofscore
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pylab as py
from pylab import figure, subplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap 
from matplotlib.gridspec import GridSpec


def custom_label(label):
    if label == "Q2":
        label = r'$Q^2\; (GeV^2)$'
    if label == "qT":
        label = r'$q_T\; (GeV)$'
    if label == "W2":
        label = r'$W^2\; (GeV^2)$'
    if label == "qToverQ":
        label = r'$q_T/Q$'
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
        label = r'$y_h$'
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
        label = r'$x_{\rm Bj}$'    


    return label

def above(dict,i,k):
      return (i-1,k) in dict.keys()

def below(dict,i,k):
      return (i+1,k) in dict.keys()

def left(dict,i,k):
      return (i,k-1) in dict.keys()

def right(dict,i,k):
      return (i,k+1) in dict.keys()

def plotEIC1(df, hadron = 'pi+', affinity = 'tmdaff', plotx = 'qT', ploty = 'z', cmap_name = 'seismic_r', yscale = 'linear'):

    data=df
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
    fig = figure(figsize=(ncols*3.2,nrows*3.2))

    # gs = gridspec.GridSpec(nrows,ncols)
    gs = GridSpec(nrows,ncols)
    gs.update(wspace=0.,hspace=0,left=0.12, right=0.86,bottom=0.13,top=0.86)
    AX={}


    # cmap = plt.get_cmap(cmap_name) # choose cmap
    cmap = get_cmap(cmap_name) # choose cmap
 
    # add a smaller subplot to explain axes
    leftb, bottomb, widthb, heightb = [0.3, 0.6, 0.15, 0.2]
    
    ax2 = fig.add_axes([leftb, bottomb, widthb, heightb])
    
    for k in sorted(bins):
        ir,ic=k
        #print k
        # ax = py.subplot(gs[ir,ic])
        ax = subplot(gs[ir,ic])
        ax.set_xlim(0,8)
        ax.set_ylim(0,1)
        #ax.set_xlim(0,data.qT.max())
        if ploty == 'z': 
            ax.set_xlim(0,1) # z is in [0,1]
            ax2.set_xlim(0,1)
            ax2.set_xlabel(r'$z_h$', fontsize=70) 
        if plotx == 'pT': 
            ax.set_ylim(0,8) # pT is in [0,2]
            ax2.set_ylim(0,8)
            ax2.set_ylabel(r'$P_T \; \rm (GeV)$', fontsize=70) 
        if plotx == 'qT': 
            ax.set_ylim(0,15) #(0,data.qT.max())
            ax2.set_ylim(0,15)
            ax2.set_ylabel(r'$q_T \; \rm (GeV)$', fontsize=70)
            
                     
            
        ax.set_yscale(yscale) # log or linear
        ax2.set_yscale(yscale)
        
        # Plot 5 ticks on x and y axis and drop the first and the last ones to avoid overlay:
        # xticks = np.round(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],5),1)[1:4]
        # yticks = np.round(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],5),1)[1:4]
        xticks = round(linspace(ax.get_xlim()[0],ax.get_xlim()[1],5),1)[1:4]
        yticks = round(linspace(ax.get_ylim()[0],ax.get_ylim()[1],5),1)[1:4]
        
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_xticklabels(xticks, fontsize=60)  
        ax2.set_yticklabels(yticks, fontsize=60)
        
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
            plot = ax.scatter(dd[ploty],dd[plotx], s=1500*dd[affinity]**0.2+20, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
            ax.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
            #ax.text(0, 2, k, fontsize=18) # show what bin is shown
            if k == (3,9):
                ax2.scatter(dd[ploty],dd[plotx], s=3500*dd[affinity]**0.2+20, c=dd[affinity], 
                                  cmap=cmap, alpha=0.8,vmin=0,vmax=1,label='') 
                ax2.plot(dd[ploty],dd[plotx],'k-', alpha=0.25,label='')
                ax.annotate('',xy=(0.,1),xycoords='axes fraction',xytext=(-1.8,2), 
                            arrowprops=dict(arrowstyle="->, head_width=1, head_length=2", color='k',lw=4))
                  
                
                
        ax.tick_params(axis='both', which='major', labelsize=25, direction='in')
        
        
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

            ax.annotate(r'$Q^2~({\rm GeV}^2)$', 
                        xy=(-1.5,5),
                        xycoords='axes fraction',
                        size=80,
                        rotation=90)

            ax.annotate(r'$x_{\rm Bj}$', 
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
                label1 = 'Central'
            elif affinity.startswith('highorder'): 
                label1 = 'High order'
            elif affinity.startswith('match'): 
                label1 = 'Matching'
            elif affinity.startswith('unclassified'): 
                label1 = 'Unclassified'

            #msg=r'${\rm %s~region~EIC~%s}$'%(label1,hadron)
            msg=r'${\rm %s~region~EIC}$'%(label1)
            ax.text(0,9.2,msg,transform=ax.transAxes,size=80)
            msg =r'${\sqrt{s}=140 \; \; \rm GeV}$'
            ax.text(0,8.2,msg,transform=ax.transAxes,size=80)
            #msg =r'${\rm %s~vs.~%s}$'%(ploty,plotx)
            #ax.text(0,5.2,msg,transform=ax.transAxes,size=80)
            
            # plot the legend of axes
            ax.legend(bbox_to_anchor=[3, -2.5], loc='center',fontsize=30,frameon=0\
                   ,markerscale=2,handletextpad=0.1)
    

    
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.5])
    cbar = fig.colorbar(plot,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=40)
   
    print(AX)
   # outname = 'EIC_%s_vs_%s_%s_%s'%(ploty,plotx,hadron,affinity)
    #py.savefig('Figs/%s.pdf'%outname)    