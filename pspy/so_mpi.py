#--
# mpi.py
# --
# this module contains hooks for parallel computing with the standard Message Passing Interface (MPI).
#
# variables and functions
#     * rank       = index of this node.
#     * size       = total number of nodes.
#     * barrier()  = halt execution until all nodes have called barrier().
#     * finalize() = terminate the MPI session (otherwise if one node finishes before the others they may be killed as well).
#
from __future__ import absolute_import, print_function
from pspy import so_misc
import warnings
import sys
import numpy as np

# blank function template
def _pass():
    pass

# set the default value
_initialized = False
_switch      = False
rank    = 0
size    = 1
comm    = None
barrier = _pass
broadcast = _pass

def init(switch = False):
    ''' initialize MPI set-up '''
    global _initialized, _switch
    global rank, size, comm
    global barrier, finalize

    exit_code = 0

    if isinstance(switch, str):
        switch = so_misc.str2bool(switch)
    else:
        pass

    if not _initialized:
        _initialized = True
    else:
        print("MPI is already intialized")
        return exit_code 

    if not switch:
        print("WARNING: MPI is turned off by default. Use mpi.init(switch=True) to initialize MPI")
        print("MPI is turned off")
        return exit_code
    else:
        _switch = True

    try:        
        from  mpi4py import MPI
        comm    = MPI.COMM_WORLD
        rank    = comm.Get_rank()
        size    = comm.Get_size()
        barrier = comm.Barrier
        print("MPI: rank %d is initalized" %rank)

    except ImportError as exc:
        sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + "). Could not load mpi4py. MPI will not be used.\n")

def is_initialized():
    global _initialized
    return _initialized

def is_mpion():
    global _switch
    return _switch

def taskrange(imax, imin = 0, shift = 0):
    '''
        Given the range of tasks, it returns the subrange of the tasks which the current processor have to perform.
        
            -- input
            -ntask: the number of tasks
            -shift: manual shift to the range
            
            -- output
            - range
        
        For size 2, rank 1
        e.g) taskrange(0, 9, offset = 0) -> [5,6,7,8,9]
        e.g) taskrange(0, 9, offset = 2) -> [7,8,9,10,11]

    '''
    global rank, size

    if not isinstance(imin, int) or not isinstance(imax, int) or not isinstance(shift, int):
        raise TypeError("imin, imax and shift must be integers")
    elif not is_initialized():
        print("MPI is not yet properly initialized. Are you sure this is what you want to do?")
    else:
        pass

    ntask = imax - imin + 1

    subrange = None
    if ntask <= 0 :
        print("number of task can't be zero")
        subrange = np.arange(0,0) # return zero range
    else:
        if size > ntask:
            delta     = 1
            remainder = 0
        else:
            delta     = ntask // size
            remainder = ntask % size

        # correction for remainder 
        start      = imin + rank*delta
        scorr      = min(rank, remainder)
        start      += scorr

        delta      += 1 if rank < remainder else 0

        end        = start + delta
        end        = min(end, imax+1)

        print("rank: %d size: %d ntask: %d start: %d end: %d" %(rank, size, ntask, start, end-1))

        subrange   = np.arange(start, end)

    return subrange


def transfer_data(data, tag, dest=0, mode='append'):
    if not is_initialized():
        raise ValueError("mpi is yet initalized")
    elif not is_mpion():
        raise ValueError("mpi is not on")
    else:
        pass

    global rank, size, comm

    senders = range(0,size)
    senders.pop(dest)

    if rank != dest:
        print("rank%d is sending data to rank%d with tag%d" %(rank, dest, tag))
        comm.send(data, dest, tag=tag)
    else:
        for sender in senders:
            print("rank%d is receiving tag%d data from rank%d" %(dest,tag,sender))
            if type(data) == dict:
                recv_data = comm.recv(source=sender, tag=tag)
                so_misc.merge_dict(data,recv_data)
            elif type(data) == np.ndarray:
                recv_data = comm.recv(source=sender, tag=tag)
                if mode == 'append':
                    data = np.append(data,recv_data)
                elif mode == 'add':
                    data += recv_data
                else:
                    assert(0)
            else:
                raise NotImplementedError()

    # wait for the end of the data transfer
    comm.Barrier()
    return data

def broadcast(data, root=0):
    comm.bcast(data, root=root)
    comm.Barrier()

def gather_set_or_dict(rank_obj, allgather=True, root=0, overlap_allowed=True):
    """Gather a set or dict from many ranks to one rank or all ranks.

    Parameters
    ----------
    rank_obj : set or dict
        The object to be gathered from each rank.
    allgather : bool, optional
        Whether the objects are gathered to all ranks or one rank, by default
        True (all ranks).
    root : int, optional
        If not allgather, the rank to which all objects are gathered, by default
        0.
    overlap_allowed : bool, optional
        If items/keys in the gathered sets/dicts are enforced to be unique
        across different ranks, by default True.

    Returns
    -------
    set or dict (or None)
        The merged rank_obj objects from each rank if allgather. Otherwise, 
        only rank root returns a set or dict, while the others return None.
    """
    if allgather:
        list_of_rank_objs = comm.allgather(rank_obj)
        gather_condition = True
    else:
        list_of_rank_objs = comm.gather(rank_obj, root=root)
        gather_condition = rank == root

    if gather_condition:
        if isinstance(rank_obj, set):
            all_obj = set()
        elif isinstance(rank_obj, dict):
            all_obj = {}
        for i, o in enumerate(list_of_rank_objs):
            if not overlap_allowed:
                if isinstance(rank_obj, set):
                    overlap = all_obj & o
                elif isinstance(rank_obj, dict):
                    overlap = all_obj.keys() & o.keys()
                assert len(overlap) == 0, \
                    f'Items in rank {i} already provided in lower rank task'
            all_obj.update(o)
    else:
        all_obj = None

    return all_obj























    



