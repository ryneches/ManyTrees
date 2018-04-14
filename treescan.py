#!/usr/bin/env python
import subprocess
from pyprind import ProgBar
from random import uniform
from os import mkdir, makedirs
from os.path import exists, join
from SuchTree import SuchTree, SuchLinkedTrees
import pandas
from numpy import linspace, zeros
from numpy.linalg import eigvalsh
from scipy.stats import skew, entropy, gaussian_kde, kurtosis, pearsonr
import matplotlib
matplotlib.use('Agg')
import seaborn
seaborn.set()
from rpy2 import robjects
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings( 'ignore', category=RRuntimeWarning )
import argparse
import json

# java options
#java_ops = [ '-XX:+UseRTMLocking', '-XX:+AggressiveOpts', '-XX:ConcGCThreads=1' ]
java_ops = [ '-XX:+AggressiveOpts', '-XX:ConcGCThreads=1' ]


# set up argument parser
parser = argparse.ArgumentParser(
            description='Simulate interactions with JPRiME.')

parser.add_argument( '--config',
                     action   = 'store',
                     dest     = 'config',
                     required = False,
                     help     = 'run config file, JSON' )

parser.add_argument( '--prefix',
                     action   = 'store',
                     dest     = 'prefix',
                     required = False,
                     help     = 'prefix name for simulations' )

parser.set_defaults( prefix = 'test' )

parser.add_argument( '--runs',
                     action   = 'store',
                     dest     = 'N',
                     type     = int,
                     required = False,
                     help     = 'number of simulations to run' )

parser.set_defaults( N = 10 )

parser.add_argument( '--start-at-run',
                     action   = 'store',
                     dest     = 'startN',
                     type     = int,
                     required = False,
                     help     = 'enumerate runs from N' )

parser.set_defaults( startN = 0 )

args = parser.parse_args()

# tell R to be quiet
robjects.r( 'options( warn = -1 )' )
robjects.r( 'sink( "/dev/null" )' )

# load libraries into the R global context
robjects.r( 'library("phytools")' )
robjects.r( 'library("igraph")' )

class JPrIMEError(Exception) :
    pass

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
             switch_rate=0.05,
             k=2.0,
             theta=0.5) :
    '''
    Time interval is always 10 units.
    '''
    
    # make output directory
    if not exists( prefix ) :
        mkdir( prefix )
   
    # build the host tree
    E = subprocess.call( [ 'java' ] + java_ops + [ '-jar', 'jprime.jar',
                           'HostTreeGen', '-bi',
                           '-min', str(min_host_leafs),
                           '-max', str(max_host_leafs),
                           '10',
                           str(birth_rate),
                           str(death_rate),
                           prefix + '/' + 'host' ] )
    
    if not E == 0 : raise JPrIMEError( 'HostTreeGen failed.' )
    
    E = subprocess.call( [ 'java' ] + java_ops + [ '-jar', 'jprime.jar',
                           'BranchRelaxer',
                           '-o', prefix + '/' + 'host.relaxed.tree',
                           prefix + '/' + 'host.pruned.tree',
                           'IIDGamma', str(k), str(theta) ] )
    
    if not E == 0 : raise JPrIMEError( 'BranchRelaxer failed on host tree.' )
    
    # build the guest tree
    E = subprocess.call( [ 'java' ] + java_ops + [ '-jar', 'jprime.jar',
                           'GuestTreeGen',
                           '-min', str(min_guest_leafs),
                           '-max', str(max_guest_leafs),
                           prefix + '/' + 'host.pruned.tree',
                           str(duplication_rate),
                           str(loss_rate),
                           str(switch_rate),
                           prefix + '/' + 'guest' ] )
    
    if not E == 0 : raise JPrIMEError( 'GuestTreGen failed.' )
    
    E = subprocess.call( [ 'java' ] + java_ops + [ '-jar', 'jprime.jar',
                           'BranchRelaxer',
                           '-o', prefix + '/' + 'guest.relaxed.tree',
                           prefix + '/' + 'guest.pruned.tree',
                           'IIDGamma', str(k), str(theta) ] )
    
    if not E == 0 : raise JPrIMEError( 'BranchRelaxer failed on guest tree.' )
    
    # load the trees
    T1 = SuchTree( prefix + '/' + 'host.relaxed.tree' )
    T2 = SuchTree( prefix + '/' + 'guest.relaxed.tree' )
    
    # populate the link matrix using the leaf names
    l = zeros( ( T1.n_leafs, T2.n_leafs ), dtype=int )
    
    hostnames = T1.leafs.keys()
    guestnames = T2.leafs.keys()
    
    for L in T2.leafs.keys() :
        guest, host = L.split('_')
        #host = 'H' + host
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
    fig.savefig( prefix + '/' + 'adjacency.png', size=6 )
    fig.clf()
    
    # plot cophylogeny using R
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
                             prefix + '/' + 'host.relaxed.tree' )
    r_code = r_code.replace( 'GUEST_TREE',
                             prefix + '/' + 'guest.relaxed.tree' )
    r_code = r_code.replace( 'LINKS',
                             prefix + '/' + 'links.csv' )
    r_code = r_code.replace( 'OUTFILE',
                             prefix + '/' + 'cophylo.pdf' )
    robjects.r( r_code )

    # calculate spectral densities
    lambdas = SLT.spectrum()
    
    a_lambd = eigvalsh( SLT.TreeA.laplacian()['laplacian'] )
    b_lambd = eigvalsh( SLT.TreeB.laplacian()['laplacian'] )
    
    with open( prefix + '/' + 'eigenvalues.csv', 'w' ) as f :
        f.write( 'graph ' + ','.join( map( str, lambdas ) ) + '\n' )
        f.write( 'TreeA ' + ','.join( map( str, a_lambd ) ) + '\n' )
        f.write( 'TreeB ' + ','.join( map( str, b_lambd ) ) + '\n' )

    bandwidth = 0.4
    X = linspace( -0.5, 1.5, 200 )
    density = gaussian_kde( lambdas/max(lambdas),
                            bw_method=bandwidth ).pdf( X )
    a_dnsty = gaussian_kde( a_lambd/max(a_lambd),
                            bw_method=bandwidth ).pdf( X )
    b_dnsty = gaussian_kde( b_lambd/max(b_lambd),
                            bw_method=bandwidth ).pdf( X )
    
    with open( prefix + '/' + 'densities.txt', 'w' ) as f :
        f.write( 'graph ' + ','.join( map( str, density ) ) + '\n' )
        f.write( 'TreeA ' + ','.join( map( str, a_dnsty ) ) + '\n' )
        f.write( 'TreeB ' + ','.join( map( str, b_dnsty ) ) + '\n' )

    # calculate Hommola correlation
    d = SLT.linked_distances()
    r,p = pearsonr( d['TreeA'], d['TreeB'] )
    
    with open( prefix + '/' + 'distances.txt', 'w' ) as f :
        f.write('TreeA ' + ','.join( map( str, d['TreeA'] ) ) + '\n' )
        f.write('TreeB ' + ','.join( map( str, d['TreeB'] ) ) +'\n' )
    
    # save jointplot of patristic distances
    jp = seaborn.jointplot( d['TreeA'], d['TreeB'], size=6 )
    jp.savefig( prefix + '/' + 'correlation.png' )
    jp.fig.clf()
    
    # output moment data
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

    moments['r']           = r
    moments['p']           = p

    with open( prefix + '/' + 'moments.csv', 'w' ) as f :
        f.write( ','.join( moments.keys()        ) + '\n' )
        f.write( ','.join( map( str, moments.values() ) ) )

    # output simulation parameters
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
    data['k']                = k
    data['theta']            = theta

    with open( prefix + '/' + 'data.csv', 'w' ) as f :
        f.write( ','.join( data.keys()        ) + '\n' )
        f.write( ','.join( map( str, data.values() ) ) )

p = ProgBar( args.N, title='simulating trees...',
             monitor=True, width=30 )
p.update()

if args.config :
    config = json.load( open( args.config ) )

    # host tree parameters
    run_dir        = config['run_dir']
    N              = config['N']
    birth_rate     = config['birth_rate']
    death_rate     = config['death_rate']
    min_host_leafs = config['min_host_leafs']
    max_host_leafs = config['max_host_leafs']
    
    # guest tree parameters
    duplication_rate = config['duplication_rate']
    loss_rate        = config['loss_rate']
    switch_rate      = config['switch_rate']
    min_guest_leafs  = config['min_guest_leafs']
    max_guest_leafs  = config['max_guest_leafs']
    
    # branch relaxer parameters
    k                = config['k']
    theta            = config['theta']

else :
    
    run_dir = args.prefix
    
    birth_rate=0.05
    death_rate=0.01
    min_host_leafs=8
    max_host_leafs=128
    
    duplication_rate=birth_rate*2.0
    loss_rate=death_rate*2.0
    switch_rate=birth_rate
    min_guest_leafs=4
    max_guest_leafs=128
    
    k=2.0
    theta=0.5

# let the command line option override the config
if args.N :
    replicates = args.N
else :
    replicates = N

if not args.startN :
    startN = 0
else :
    startN = int(args.startN)

for i in range( args.startN, replicates + args.startN ) :
    
    #duplication_rate = uniform( 0.05, 0.35 )
    #loss_rate        = uniform( 0.0, 0.25 )
    #switch_rate      = uniform( 0.0, 0.075 )
    
    if not exists( run_dir ) :
        makedirs( run_dir )
    prefix = join( run_dir, 'run' )
    
    # log simulation parameters
    with open( prefix + '.log', 'a' ) as f :
        f.write( prefix + str(i) + ' :\n' )
        f.write( '   birth_rate       = ' + str(birth_rate) + '\n' )
        f.write( '   death-rate       = ' + str(death_rate) + '\n' )
        f.write( '   min_host_leafs   = ' + str(min_host_leafs) + '\n' )
        f.write( '   max_host_leafs   = ' + str(max_host_leafs) + '\n' )
        f.write( '   min_guest_leafs  = ' + str(min_guest_leafs) + '\n' )
        f.write( '   max_guest_leafs  = ' + str(max_guest_leafs) + '\n' )
        f.write( '   duplication_rate = ' + str(duplication_rate) + '\n' )
        f.write( '   loss_rate        = ' + str(loss_rate) + '\n' )
        f.write( '   switch_rate      = ' + str(switch_rate) + '\n' )
        f.write( '   k                = ' + str(k) + '\n' )
        f.write( '   theta            = ' + str(theta) + '\n' )
    
    #simtree( 'test' + str(i), switch_rate = switch_rate )
    
    try :
        simtree( prefix + str(i),
                 birth_rate=birth_rate,
                 death_rate=death_rate,
                 min_host_leafs=min_host_leafs,
                 max_host_leafs=max_host_leafs,
                 min_guest_leafs=min_guest_leafs,
                 max_guest_leafs=max_guest_leafs,
                 duplication_rate=duplication_rate,
                 loss_rate=loss_rate,
                 switch_rate=switch_rate,
                 k=k,
                theta=theta )
    except JPrIMEError as jpe :
        with open( prefix + '.log', 'a' ) as f :
            f.write( '    FAILED : ' + str(jpe) + '\n' )
        with open( prefix + str(i) + '/' + 'fail.msg', 'w' ) as f :
            f.write( str(jpe) )

    p.update()
