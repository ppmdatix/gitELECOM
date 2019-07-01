from __future__ import print_function, division
from keras import backend as K
import tensorflow as tf
import numpy as np


def CVSS(naf, ns, sb, db, rs, nr, sa, nfc, mini=0.47, maxi=.8, mode="keras"):
    access_vector = (naf + 2.1) / 2.2
    attack_complexity = (ns + 2.1) * (sb + db + 3.1) / 7.2
    authentification = (rs + nr + 2.1) * (2 + sa) / 4.2
    conf_impact = (naf + 2.1) / 2.2
    int_impact = (nfc + 2.1) / 2.2
    availibility_impact = (sb + db + 2.1) / 4.2
    exploitability = 50 * access_vector * attack_complexity * authentification
    impact = 1 - (1 - conf_impact) * (1 - int_impact) * (1 - availibility_impact)
    if mode == "keras":
        return K.minimum(K.maximum(1 - ((impact * .5 + exploitability) - mini) / (maxi - mini), 0), 1)
    elif mode == "numpy":
        return np.minimum(np.maximum(1 - ((impact * .5 + exploitability) - mini) / (maxi - mini), 0), 1)


def hurting_raw(x, dico, mode="keras"):
    return CVSS(naf=x[dico["num_access_files"]],
                ns=x[dico["num_shells"]],
                sb=x[dico["src_bytes"]],
                db=x[dico["dst_bytes"]],
                rs=x[dico["root_shell"]],
                nr=x[dico["num_root"]],
                sa=x[dico["su_attempted"]],
                nfc=x[dico["num_file_creations"]], mode=mode)


def hurting(traffic, dico):
    t = tf.transpose(traffic)
    return hurting_raw(t, dico=dico)


def custom_loss(intermediate_output, dico, alpha, offset):

    def loss_fucntion(y_true, y_pred):
        L = - K.log(K.maximum((y_pred+1)*.5, 1e-9))
        hurt = hurting(intermediate_output, dico=dico)
        loss = L * hurt + (1 - hurt) * (alpha * L + offset)
        return loss

    return loss_fucntion


def loss_function_discriminator(y_true, y_pred):
    return -y_true*K.log(K.maximum((y_pred+1)*.5, 1e-9)) - (1-y_true)*K.log(K.maximum(1-(y_pred+1)*.5, 1e-9))