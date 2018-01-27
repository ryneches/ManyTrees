#!/usr/bin/env python
import subprocess
from SuchTree import SuchTree, SuchLinkedTrees
import pandas
from numpy import linspace, zeros
from numpy.linalg import eigvalsh
from scipy.stats import skew, entropy, gaussian_kde, kurtosis, pearsonr
from matplotlib import pylab
import seaborn
seaborn.set()

def pdd( a, b ) :
    '''Jensen-Shannon distance'''
    return ( 0.5 * entropy( a, b ) + 0.5 * entropy( b, a ) )**(0.5)

# build the host tree
subprocess.call( [ 'java', '-jar', 'jprime.jar',
                   'HostTreeGen', '-bi',
                   '-min', '3', '10', '0.3', '0.1',
                   'myhost' ] )

# build the guest tree
subprocess.call( [ 'java', '-jar', 'jprime.jar',
                   'GuestTreeGen', '-min', '4', '-max', '20',
                   'myhost.pruned.tree',
                   '0.2', '0.1', '0.05','myguest' ] )

# load the trees
T1 = SuchTree( 'myhost.pruned.tree' )
T2 = SuchTree( 'myguest.pruned.tree' )

# populat the link matrix using the leaf names
l = zeros( ( T1.n_leafs, T2.n_leafs ), dtype=int )

hostnames = T1.leafs.keys()
guestnames = T2.leafs.keys()

for L in T2.leafs.keys() :
    guest, host = L.split('_')
    host = 'H' + host
    i = hostnames.index( host )
    j = guestnames.index( L )
    l[i,j] = 1

links = pandas.DataFrame( l, index=hostnames, columns=guestnames )

# initialize the SuchLinkedTrees object
SLT = SuchLinkedTrees( T1, T2, links )

# plot the adjacency matrix
aj = SLT.adjacency()
pylab.figure( figsize=(5,5) )
lp_plot = pylab.pcolor( aj, cmap='viridis', vmin=0, vmax=1 )
fig = lp_plot.get_figure()
fig.savefig( 'adjacency.png' )

lambdas = SLT.spectrum()

a_lambd = eigvalsh( SLT.TreeA.laplacian()['laplacian'] )
b_lambd = eigvalsh( SLT.TreeB.laplacian()['laplacian'] )

with open( 'eigenvalues.csv', 'w' ) as f :
    f.write( ','.join( map( str, lambdas ) ) )

bandwidth = 0.4
X = linspace( -0.5,1.5,200 )
density = gaussian_kde( lambdas/max(lambdas),
                        bw_method=bandwidth ).pdf( X )
a_dnsty = gaussian_kde( a_lambd/max(a_lambd),
                        bw_method=bandwidth ).pdf( X )
b_dnsty = gaussian_kde( b_lambd/max(b_lambd),
                        bw_method=bandwidth ).pdf( X )

moments = {}
moments['eigengap']   = lambdas[-1] - lambdas[-2]
moments['skew']       = skew( density )
moments['kurtosis']   = kurtosis( density )
moments['treedist']   = pdd( a_dnsty, b_dnsty )
moments['occupancy']  = (2.0 * SLT.n_links) / ( SLT.TreeA.n_leafs + SLT.TreeB.n_leafs )
moments['squareness'] = float( SLT.TreeA.n_leafs ) / SLT.TreeB.n_leafs

with open( 'moments.csv', 'w' ) as f :
    f.write( ','.join( moments.keys()        ) + '\n' )
    f.write( ','.join( map( str, moments.values() ) ) )
