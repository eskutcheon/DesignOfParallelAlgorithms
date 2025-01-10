After login to titan.hpc.msstate.edu, you will need to login to the head node
of the scout cluster with the command:

ssh -Y scout-login


Before you use the scount-login system for the first time you will need
load in the mpi package which you can do with the commands

module load cuda/11.2.1 slurm
module save default

This will load the CUDA compiler and the slurm batch processing system
into your executable path.  The "module save" command will save this
setting so that it will apply to subsequent logins.


You can use the editors vi, nano, or emacs to perform this editing. 

*   nano is the easiest editor to use for the uninitiated.

Now you will need to copy the project3.tar file to scout-login (you only need
to do this once).  To copy this you can use the command

Then you can copy the project files to shadow from titan using the command 
(assuming you are logged into scout-login)

scp titan:/scratch/CSE4163/Project3/project3.tar .

This will copy the file from titan to your home directory.

To unpack this file just use the command:

tar xvf project3.tar


You will now have a directory called 'project3' that will contain the starting
source code for your project.  Enter the directory using the command 
'cd project3'


Then you can compile using the command "make" in the project directory.  This
assumes that you have performed the module load command discussed above.


This directory contains several program files:

fluid.cc:    This is the reference serial CPU implementation you will use
             to set the baseline running time and to collect data for
             testing your CUDA implementation

fluid.cu:    This is the CUDA version of the fluid solver.  It is your job to 
             convert the computations into CUDA kernels to run on the GPGPU.
             The initial conditions and sum kinetic energy routines have
             already been converted.

runfluid.js: This is a job script for running the baseline serial CPU version
             on the cluster

runfluid32.js: This job script will run the CUDA version of the code on the
               GPGPU cluster for the 32x32x32 sized grid.

runfluid64.js: This job script will run the CUDA version of the code on the
               GPGPU cluster for the 64x64x64 sized grid

runfluid128.js: This job script will run the CUDA version of the code on the
                GPGPU cluster for the 128x128x128 sized grid

runfluid256.js: This job script will run the CUDA version of the code on the
                GPGPU cluster for the 256x256x256 sized grid

cuda:  This is a directory that contains example CUDA code

-----------------------------------------------------------------------------

How to submit jobs to the parallel cluster using the Slurm batch system:

To submit a job once the program has been compiled use one of the
provided Slurm job scripts (these end in .js).  These job scripts have
been provided to run parallel jobs on the cluster.  To submit a job use 
the "sbatch" command. (note:  "man sbatch" to get detailed information on this 
command)

example:
sbatch runfluid32.js

To see the status of jobs on the queue, use squeue.  Example:

scout-login-1[134] lush$ squeue -u lush
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
             20298   48p160h  Work32P     lush PD       0:00      2 (ReqNodeNotAvail, UnavailableNodes:Scout-1-[01-40],Scout-2-[01-40],Scout-3-[01-40],Scout-4-[01-40]) 
             20297   48p160h  Work16P     lush PD       0:00      1 (ReqNodeNotAvail, UnavailableNodes:Scout-1-[01-40],Scout-2-[01-40],Scout-3-[01-40],Scout-4-[01-40]) 

This lists information associated with the job.  The important things to note
are the Job id and the state (ST).  The state tells the status of the job.  
Generally the status will be one of two values:  PD -  for pending

Additionally, if you decide that you don't want to run a job, or it seems to
not work as expected, you can run "qdel Job id" to delete it from the queue.
For example, to remove the above job from the queue enter the command

scancel 20297
