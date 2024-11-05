from globals import *
from run_dqn_surrogate_accelerator import run
import tensorflow as tf

if __name__ == "__main__" :
    # ----------- This part is to avoid any CUBLAS errors ------------------------------
    print( tf.config.list_physical_devices( 'GPU' ) )  # check for the GPU, if being used
    gpus = tf.config.experimental.list_physical_devices( 'GPU' )
    if gpus :
        try :
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus :
                tf.config.experimental.set_memory_growth( gpu , True )
            logical_gpus = tf.config.experimental.list_logical_devices( 'GPU' )
            print( len( gpus ) , "Physical GPUs," , len( logical_gpus ) , "Logical GPUs" )
        except RuntimeError as e :
            # Memory growth must be set before GPUs have been initialized
            print( e )
    # ----------- This part is to avoid any CUBLAS errors ------------------------------

    os.makedirs(EPISODES_PLOTS_DIR)

    run( episodes = AGENT_EPISODES ,
         nsteps = AGENT_NSTEPS ,
         in_play_mode = IN_PLAY_MODE )
