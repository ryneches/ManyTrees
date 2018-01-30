#!/usr/bin/env python
import subprocess
from pyprind import ProgBar
from random import uniform
from os import mkdir
from SuchTree import SuchTree, SuchLinkedTrees
import pandas
from numpy import linspace, zeros
from numpy.linalg import eigvalsh
from scipy.stats import skew, entropy, gaussian_kde, kurtosis, pearsonr
import seaborn
seaborn.set()
from rpy2 import robjects
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings( 'ignore', category=RRuntimeWarning )


# tell R to be quiet
robjects.r( 'options( warn = -1 )' )
robjects.r( 'sink( "/dev/null" )' )

# load libraries into the R global context
robjects.r( 'library("phytools")' )
robjects.r( 'library("igraph")' )

def pdd( a, b ) :
    '''Jensen-Shannon distance'''
    return ( 0.5 * entropy( a, b ) + 0.5 * entropy( b, a ) )**(0.5)

def simtree( prefix,
             birth_rate=0.3,
             death_rate=0.1,
             min_host_leafs=8,
             max_host_leafs=64,
             min_guest_leafs=4,
             max_guest_leafs=128,
             duplication_rate=0.2,
             loss_rate=0.1,
             switch_rate=0.05 ) :
    '''
    Time interval is always 10 units.
    '''
    
    # make output directory
    mkdir( prefix )

    # build the host tree
    r = subprocess.call( [ 'java', '-jar', 'jprime.jar',
                           'HostTreeGen', '-bi',
                           '-min', str(min_host_leafs),
                           '-max', str(max_host_leafs),
                           '10',
                           str(birth_rate),
                           str(death_rate),
                           prefix + '/' 'host' ] )
    
    if not r == 0 : raise( 'JPRiME failed, exiting' ) 

    # build the guest tree
    r = subprocess.call( [ 'java', '-jar', 'jprime.jar',
                           'GuestTreeGen',
                           '-min', str(min_guest_leafs),
                           '-max', str(max_guest_leafs),
                           prefix + '/' + 'host.pruned.tree',
                           str(duplication_rate),
                           str(loss_rate),
                           str(switch_rate),
                           prefix + '/' 'guest' ] )
   
    if not r == 0 : raise ( 'JPRiME failed, exiting' )

    # load the trees
    T1 = SuchTree( prefix + '/' + 'host.pruned.tree' )
    T2 = SuchTree( prefix + '/' + 'guest.pruned.tree' )
    
    # populate the link matrix using the leaf names
    l = zeros( ( T1.n_leafs, T2.n_leafs ), dtype=int )
    
    hostnames = T1.leafs.keys()
    guestnames = T2.leafs.keys()
    
    for L in T2.leafs.keys() :
        guest, host = L.split('_')
        host = 'H' + host
        i = hostnames.index( host )
        j = guestnames.index( L )
        l[i,j] = 1
    
    links = pandas.DataFrame( l,
                              index=hostnames,
                              columns=guestnames )
    links.to_csv( prefix + '/' + 'links.csv' )
    
    # initialize the SuchLinkedTrees object
    SLT = SuchLinkedTrees( T1, T2, links )
    
    # plot the adjacency matrix
    aj = SLT.adjacency()
    lp_plot = seaborn.heatmap( aj.T, cmap='viridis',
                               vmin=0, vmax=1, cbar=False,
                               square=True, xticklabels=False,
                               yticklabels=False )
    lp_plot.invert_yaxis()
    fig = lp_plot.get_figure()
    fig.savefig( prefix + '/' + 'adjacency.png' )
    
    # plot cophylogeny
    r_code = '''
    tr1 <- read.tree( "HOST_TREE" )
    tr2 <- read.tree( "GUEST_TREE" )
    links <- read.csv( "LINKS", row.names=1, stringsAsFactors = F )
    im <- graph_from_incidence_matrix( as.matrix( links ) )
    assoc <- as_edgelist( im )
    obj <- cophylo( tr1, tr2, assoc=assoc )
    pdf( "OUTFILE", width = 10, height = 12 )
    plot( obj )
    dev.off()
    '''
    r_code = r_code.replace( 'HOST_TREE',
                             prefix + '/' + 'host.pruned.tree' )
    r_code = r_code.replace( 'GUEST_TREE',
                             prefix + '/' + 'guest.pruned.tree' )
    r_code = r_code.replace( 'LINKS',
                             prefix + '/' + 'links.csv' )
    r_code = r_code.replace( 'OUTFILE',
                             prefix + '/' + 'cophylo.pdf' )
    robjects.r( r_code )

    # calculate moments
    lambdas = SLT.spectrum()
    
    a_lambd = eigvalsh( SLT.TreeA.laplacian()['laplacian'] )
    b_lambd = eigvalsh( SLT.TreeB.laplacian()['laplacian'] )
    
    with open( prefix + '/' + 'eigenvalues.csv', 'w' ) as f :
        f.write( ','.join( map( str, lambdas ) ) )
    
    bandwidth = 0.4
    X = linspace( -0.5, 1.5, 200 )
    density = gaussian_kde( lambdas/max(lambdas),
                            bw_method=bandwidth ).pdf( X )
    a_dnsty = gaussian_kde( a_lambd/max(a_lambd),
                            bw_method=bandwidth ).pdf( X )
    b_dnsty = gaussian_kde( b_lambd/max(b_lambd),
                            bw_method=bandwidth ).pdf( X )
    
    moments = {}
    moments['eigengap']    = lambdas[-1] - lambdas[-2]
    moments['skew']        = skew( density )
    moments['kurtosis']    = kurtosis( density )
    moments['treedist']    = pdd( a_dnsty, b_dnsty )
    moments['occupancy']   = ( 2.0 * SLT.n_links ) \
                             / ( SLT.TreeA.n_leafs \
                                 + SLT.TreeB.n_leafs )
    moments['squareness']  = float( SLT.TreeA.n_leafs ) \
                             / SLT.TreeB.n_leafs
    
    with open( prefix + '/' + 'moments.csv', 'w' ) as f :
        f.write( ','.join( moments.keys()        ) + '\n' )
        f.write( ','.join( map( str, moments.values() ) ) )

    data = {}
    data['prefix']           = prefix
    data['host_leafs']       = T1.n_leafs
    data['guest_leafs']      = T2.n_leafs
    data['links']            = SLT.n_links
    data['birth_rate']       = birth_rate
    data['death_rate']       = death_rate
    data['min_host_leafs']   = min_host_leafs
    data['max_host_leafs']   = max_host_leafs
    data['min_guest_leafs']  = min_guest_leafs
    data['max_guest_leafs']  = max_guest_leafs
    data['duplication_rate'] = duplication_rate
    data['loss_rate']        = loss_rate
    data['switch_rate']      = switch_rate
    
    with open( prefix + '/' + 'data.csv', 'w' ) as f :
        f.write( ','.join( data.keys()        ) + '\n' )
        f.write( ','.join( map( str, data.values() ) ) )

p = ProgBar( 10, title='building trees...' )
p.update()
for i in range(10) :
    duplication_rate = uniform( 0.25, 0.35 )
    loss_rate        = uniform( 0.15, 0.25 )
    switch_rate      = uniform( 0.0, 0.075 )
    simtree( 'test' + str(i), switch_rate = switch_rate )
    p.update()