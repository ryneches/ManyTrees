
Java :
* `JPRiME` (provided)

R :
* `phytools`
* `igraph`

Python :
* `SuchTree`
* `pandas`
* `numpy`
* `scipy`
* `matplotlib`
* `seaborn`
* `pyprind`
* `dendropy`

### Simulations

Parameters :
* switching
  * none : 0
  * slow : =death_rate 
  * fast : =birth_rate
* guest_evolution
  * slow : duplication_rate = 0.25 * birth_rate, loss_rate = 0.25 * death_rate
  * balanced : duplication_rate = birth_rate, loss_rate = death_rate
  * fast : duplication_rate = 4 * birth_rate, loss_rate = 4 * death_rate

Simulation sets :

|                          | no switching        | slow switching        | fast switching        |
| ------------------------ | ------------------- | --------------------- | --------------------- |
| slow guest evolution     | `noswitch_slow`     | `slowswith_slow`      | `fastswitch_slow`     |
| balanced guest evolution | `noswitch_balanced` | `slowswitch_balanced` | `fastswitch_balanced` |
| fast guest evolution     | `noswitch_fast`     | `slowswitch_fast`     | `fastswitch_fast`     |
