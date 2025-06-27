import numpy as np
import pandas as pd
import ultraplot as upt 

def equalareaplot(dec=None,inc=None,a95=None, legendbool=True,
                  cirlabel=None,circolor='k',
                  fisher=False,fisherdf=None,fishertextloc=None,
                  ax=None,
                  type=0,line=True,showticks=False,markersize=4,linewidth=1,steptext=None,
                  starlabel=None,starcolor='k',):
    """Create an equal-area plot.
       inc>=0:down k
        else:up w
    """
    if dec is not None:
        if isinstance(dec, (int, float)):
            dec = [dec]
        elif not isinstance(dec, list):
            dec = list(dec)
    if inc is not None:
        if isinstance(inc, (int, float)):
            inc = [inc]
        elif not isinstance(inc, list):
            inc = list(inc)
    if a95 is not None:
        if isinstance(a95, (int, float)):
            a95 = [a95]
        elif not isinstance(a95, list):
            a95 = list(a95)
    if ax is None:
        fig, ax = upt.subplots(figsize=(18/2.54, 18/2.54),
        top='1.3cm',
        bottom='0.8cm',
        right='0.5cm',
        left='0.5cm',
         ncols=1, nrows=1,share=False,proj="polar", )
    else:
        pass
    if dec is None or inc is None:
        pass
    else:
        dec_c=np.radians(dec)
        inc_c=EqualArea(inc)
        downbool = np.array(inc) >= 0
        upbool = np.array(inc) < 0
        downc= None
        upc=None
        try:
            downc=ax.plot(dec_c[downbool],inc_c[downbool],'o',facecolor=circolor,edgecolor=circolor, markersize=markersize, label='Down',zorder=100,clip_on=False)
        except:
            pass
        try:
            upc=ax.plot(dec_c[upbool],inc_c[upbool],'o',facecolor='w', edgecolor=circolor,markersize=markersize, label='Up',zorder=100,clip_on=False)
        except:
            pass
        if line:
            ax.plot(dec_c,inc_c,'k--',linewidth=linewidth) 
        if steptext is not None:
            pass
    if type == 0:
        angles = [0, 90, 180, 270]
        angle_ranges = [(10, 100, 10)] + [(10, 90, 10)] * 3 
        symbols = []
        for i, (angle, (start, end, step)) in enumerate(zip(angles, angle_ranges)):
            Xsym, Ysym = [], []
            for Pl in range(start, end, step):
                ix = EqualArea(Pl)
                Xsym.append(np.radians(angle))
                Ysym.append(ix)
            symbol = ax.plot(Xsym, Ysym, marker='+', color='gray', markersize=markersize+2, linestyle='None')
            symbols.append(symbol)
        list1=None
        list2=None
        grid=False
        ax.set_rticks([])
    elif type == 1:
        list1=[EqualArea(x) for x in range(10,90,20)]
        if showticks:
            list2=[str(x) for x in range(10,90,20)]
        else:
            list2="null"#[str(x) for x in range(10,90,20)]
        grid=True

    if a95 is not None:
        for i, a95_value in enumerate(a95):
            if a95_value is not None:
                dec_i = dec[i]
                inc_i = inc[i]
                
                xpos, ypos, xneg, yneg = alphacirc(dec_i, inc_i, alpha=a95_value)
                
                try:
                    ax.plot(xpos, ypos, 'k-', linewidth=0.5,alpha=0.7)
                except:
                    pass
                try:

                    ax.plot(xneg, yneg, 'k-', linewidth=0.5,alpha=0.7)
                except:
                    pass
    if fisher:
        # Plot Fisher mean direction
        if fisherdf is not None:
            fisherresualt=fisherdf
            try:
                dec_f, inc_f,alpha95 = fisherresualt.specimen_dec[0], fisherresualt.specimen_inc[0], fisherresualt.specimen_alpha95[0]
            except:
                dec_f, inc_f,alpha95 = fisherresualt.dec[0], fisherresualt.inc[0], fisherresualt.alpha95[0]
            
        else:
            try:
                fisherresualt =fisher_mean(dec, inc)
                dec_f, inc_f,alpha95 = fisherresualt.dec[0], fisherresualt.inc[0], fisherresualt.alpha95[0]
            except:
                pass
        downS = None
        upperS = None
        if inc_f < 0:
            #Up direction
            # upperS=ax.plot(np.radians(dec_f), EqualArea(inc_f), '*',facecolor='w', markersize=markersize+3,markeredgewidth=0.2, edgecolor=starcolor,clip_on=False,zorder=1001,label='Fisher Mean Up',)
            upperS = ax.scatter(np.radians(dec_f), EqualArea(inc_f), marker='*', 
                facecolors='w', edgecolors=starcolor, s=(markersize+3)**2,
                linewidths=0.2, clip_on=False, zorder=10001, 
                label='Fisher Mean Up')
        else:
            #Down direction
            #downS=ax.plot(np.radians(dec_f), EqualArea(inc_f), '*',facecolor=starcolor, markersize=markersize+3,markeredgewidth=0.2,edgecolor=starcolor, clip_on=False,zorder=1001,label='Fisher Mean Down',)
            downS = ax.scatter(np.radians(dec_f), EqualArea(inc_f), marker='*', 
                facecolors=starcolor, edgecolors=starcolor, 
                s=(markersize+3)**2, linewidths=0.2, clip_on=False, 
                zorder=10001, label='Fisher Mean Down')
        
        fxpos, fypos, fxneg, fyneg = alphacirc(dec_f, inc_f, alpha=alpha95)
        ax.fill(fxpos, fypos, color='gray', alpha=0.5,zorder=10000,)
        ax.fill(fxneg, fyneg, color='gray', alpha=0.5,zorder=10000)
        if starlabel is not None and (downS is not None or upperS is not None):
            handles = []
            if downS is not None:

                handles.append(downS)
            if upperS is not None:

                handles.append(upperS)

            if handles:
                Slegend=ax.legend(handles=handles, label=starlabel,loc="t", ncols=2, frame=False)
        else:
            pass

        if fishertextloc is None:
            textf=False
        elif fishertextloc == 't':
            textf=True
            textx=0.5
            texty=0.8
        elif fishertextloc == 'b':
            textf=True
            textx=0.5
            texty=0.2
        elif fishertextloc == 'l':
            textf=True
            textx=0.2
            texty=0.5
        elif fishertextloc == 'r':
            textf=True
            textx=0.8
            texty=0.5
        else:
            textf=False
        if textf:
            nt='n='+str(int(fisherresualt.n.values))
            Dt='D='+str(round(float(fisherresualt.dec.values),2))
            It='I='+str(round(float(fisherresualt.inc.values),2))
            a95t='α_{95}='+str(round(float(fisherresualt.alpha95.values),2))
            kt='κ='+str(round(float(fisherresualt.k.values),2))
            alltext= r'${}${}${}${}${}${}${}${}${}$'.format(nt,'\n',Dt,'\n',It,'\n',a95t,'\n',kt)
            alltextP=ax.text(textx,texty,alltext,ha='center', va='center',color='k',fontsize=12,transform=ax.transAxes,clip_on=False)
            alltextP.set_bbox(dict(facecolor='w',linewidth=0, alpha=0.5, pad=0)) 
        else:
            pass
    else:
        pass
    downl=ax.scatter(0, 100, marker='o',markersize=markersize+10,color='k', edgecolor='k',label='Down')
    upperl=ax.scatter(0, 100, marker='o',markersize=markersize+10,color='w', edgecolor='k',label='Up') 

    if cirlabel is not None:
        llegend=ax.legend(handles=[downc, upc], label=cirlabel,loc="t", ncols=2, frame=False)
    else:
        if legendbool:
            plegend=ax.legend(handles=[downl, upperl],loc="t", ncols=2, frame=False)
        else:
            pass
    ax.format(
        thetadir=-1,
        thetalines=90,
        theta0="N",
        rlim=(90.,0.),
        grid=grid,
        rlines=list1,
        rformatter=list2,
    )

    return ax


def EqualArea(Pl):
    Pl_array = np.atleast_1d(Pl)
    Pl_adjusted = np.abs(Pl_array)
    result =90- np.sqrt(2.) * 90. * np.sin(np.radians(90. - Pl_adjusted) / 2.)

    
    if np.isscalar(Pl):
        return float(result[0])
    else:
        return result

def alphacirc(c_dec,c_inc,alpha):
    
    alpha=np.radians(alpha) 
    t=np.zeros((3,3)) 
    t[2]=dir2cart([c_dec,c_inc])
    plane1=[c_dec,c_inc-90.]
    plane2=[c_dec+90.,0]
    
    t[0]=dir2cart(plane1)
    t[1]=dir2cart(plane2)
    t=t.transpose()
    npts=201
    xnum=float(npts-1.)/2.
    v=[0,0,0]
    PTS=[]
    for i in range(npts): 
            psi=float(i)*np.pi/xnum
            v[0]=np.sin(alpha)*np.cos(psi)
            v[1]=np.sin(alpha)*np.sin(psi)
            if alpha==np.pi/2.:
                v[2]=0.
            else:
                v[2]=np.sqrt(1.-v[0]**2 - v[1]**2)
            elli=[0,0,0]
            for j in range(3):
                for k in range(3):
                    elli[j]=elli[j] + t[j][k]*v[k] 
            PTS.append(cart2dir(elli))
    pts_array = np.array(PTS)
    upper_mask = pts_array[:, 1] > 0
    lower_mask = ~upper_mask

    # Upper hemisphere (negative inclination)
    xpos = np.radians(pts_array[upper_mask, 0]).tolist()
    ypos = EqualArea(-pts_array[upper_mask, 1]).tolist()

    # Lower hemisphere (positive inclination)
    xneg = np.radians(pts_array[lower_mask, 0]).tolist()
    yneg = EqualArea(pts_array[lower_mask, 1]).tolist()
    if len(xpos) > 0 and len(xneg) > 0:
        rxpos=xpos[::-1]
        
        half = len(xneg) // 2
        xneg = xneg[half:] + xneg[:half]
        yneg = yneg[half:] + yneg[:half]
        rxneg=xneg[::-1]
        rypos=[0.0] * len(xpos)
        ryneg=[0.0] * len(xneg)
        xpos.extend(rxpos)
        ypos.extend(rypos)
        xneg.extend(rxneg)
        yneg.extend(ryneg)
    else:
        pass

    return xpos, ypos, xneg, yneg

def fisher_mean(dec=None, inc=None, di_block=None, unit_vector=True):
    if di_block is None:
        if dec is None or inc is None:
            raise ValueError("Must provide either di_block or both dec and inc")
        di_block = []
        if unit_vector:
            for n in range(len(dec)):
                di_block.append([dec[n], inc[n], 1.0])
        else:
            for n in range(len(dec)):
                di_block.append([dec[n], inc[n]])
    N = len(di_block)
    fpars = {}
    if N < 2: 
        return {'dec': di_block[0], 
                'inc': di_block[1]}
    X = []
    for direction in di_block:
        cart_coords = dir2cart(direction)
        X.append(cart_coords)
    X = np.array(X)

    Xbar = X.sum(axis=0)

    R = np.linalg.norm(Xbar)
    Xbar = Xbar / R
    direction = cart2dir(Xbar)
    # print(direction)
    try:
        fpars["dec"] = direction[0][0]
        fpars["inc"] = direction[0][1]
    except:
        fpars["dec"] = direction[0]
        fpars["inc"] = direction[1]
    fpars["n"] = N
    fpars["r"] = R
    
    if N != R:
        k = (N - 1.) / (N - R)
        fpars["k"] = k
        csd = 81. / np.sqrt(k)
    else:
        fpars['k'] = 'inf'
        csd = 0.
        
    b = 20.**(1. / (N - 1.)) - 1
    a = 1 - b * (N - R) / R
    if a < -1:
        a = -1
    a95 = np.degrees(np.arccos(a))
    fpars["alpha95"] = a95
    fpars["csd"] = csd
    if a < 0:
        fpars["alpha95"] = 180.0

    return pd.DataFrame(fpars, index=[0])

def dir2cart(d):
    """
    Converts a list or array of vector directions in degrees (declination,
    inclination) to an array of the direction in cartesian coordinates (x,y,z).
    """
    rad = np.pi/180.
    ints = np.ones(len(d)).transpose()  # get an array of ones to plug into dec,inc pairs
    d = np.array(d).astype('float')
    if len(d.shape) > 1:  # array of vectors
        decs, incs = d[:, 0] * rad, d[:, 1] * rad
        if d.shape[1] == 3:
            ints = d[:, 2]  # take the given lengths
    else:  # single vector
        decs, incs = np.array(float(d[0])) * rad, np.array(float(d[1])) * rad
        if len(d) == 3:
            ints = np.array(d[2])
        else:
            ints = np.array([1.])
    cart = np.array([ints * np.cos(decs) * np.cos(incs), ints *
                     np.sin(decs) * np.cos(incs), ints * np.sin(incs)])
    cart = np.array([ints * np.cos(decs) * np.cos(incs), ints *
                     np.sin(decs) * np.cos(incs), ints * np.sin(incs)]).transpose()
    return cart

def cart2dir(cart):
    """
    Converts a direction in cartesian coordinates into declinations and inclination.
    """
    cart = np.array(cart)
    rad = np.pi/180.  # constant to convert degrees to radians
    if len(cart.shape) > 1:
        Xs, Ys, Zs = cart[:, 0], cart[:, 1], cart[:, 2]
    else:  # single vector
        Xs, Ys, Zs = cart[0], cart[1], cart[2]
    if np.iscomplexobj(Xs):
        Xs = Xs.real
    if np.iscomplexobj(Ys):
        Ys = Ys.real
    if np.iscomplexobj(Zs):
        Zs = Zs.real
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)  # calculate resultant vector length
    # calculate declination taking care of correct quadrants (arctan2) and
    # making modulo 360.
    Decs = (np.arctan2(Ys, Xs) / rad) % 360.
    try:
        # calculate inclination (converting to degrees) #
        Incs = np.arcsin(Zs / Rs) / rad
    except:
        print('trouble in cart2dir')  # most likely division by zero somewhere
        return np.zeros(3)
    
    direction_array = np.array([Decs, Incs, Rs]).transpose()  # directions list
    
    return direction_array

def EQsteptext(ax, dec, inc, steptext, step=3):
    steps = []
    treatmtext =  steptext

    
    # Create array of valid indices (not NaN in x or y)
    x_array = np.array(dec)
    y_array = np.array(inc)
    valid_mask = ~np.isnan(x_array) & ~np.isnan(y_array)
    valid_indices = np.where(valid_mask)[0]
    xtext = x_array[::2]
    ytext = y_array[::2]
    steptextplot=np.array(steptext.values).flatten().tolist()
    steptextplot=steptextplot[::2]
    ax.text(xtext, ytext, steptextplot,
            ha='left', va='bottom', color='gray', fontsize=6, clip_on=False)