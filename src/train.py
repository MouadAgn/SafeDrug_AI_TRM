import os
import tensorflow as tf
import numpy as np
from data_loader import get_bpRNA_data, RNA_MAP, STRUCT_MAP, MAX_LEN
from model import TRM_Official

# --- Paramètres Optimisés ---
D_MODEL = 64
N_SUP = 3       # Deep Supervision steps (réduit un peu pour la vitesse)
EPOCHS = 5
BATCH_SIZE = 16 

def train():
    # 1. Données
    X_train, Y_train = get_bpRNA_data(limit=2000) 
    
    # Dataset optimisé
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.shuffle(500).batch(BATCH_SIZE, drop_remainder=True)
    
    # 2. Modèle
    model = TRM_Official(
        input_vocab_size=len(RNA_MAP),
        output_vocab_size=len(STRUCT_MAP),
        d=D_MODEL,
        max_len=MAX_LEN,
        n=2, 
        T=2 
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Reduction=NONE est crucial pour pouvoir appliquer le masque manuellement
    loss_fn_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    print(f"--- Démarrage TRM (Mode: Masked Training) ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        step = 0
        
        for x_batch, y_true in dataset:
            batch_len = tf.shape(x_batch)[0]
            
            # Initialisation
            y_curr_tokens = tf.zeros((batch_len, MAX_LEN), dtype=tf.int32)
            y_curr_emb = model.emb_y(y_curr_tokens)
            z_curr = tf.zeros((batch_len, MAX_LEN, D_MODEL))
            x_emb = model.emb_x(x_batch) + model.pos_emb[:, :MAX_LEN, :]

            # CRÉATION DU MASQUE (Ignorer les 0 dans le calcul d'erreur)
            # mask est True là où il y a de la donnée, False là où c'est du padding (0)
            mask = tf.math.not_equal(y_true, 0)
            mask = tf.cast(mask, dtype=tf.float32)

            with tf.GradientTape() as tape:
                batch_loss = 0
                
                # --- BOUCLE DEEP SUPERVISION ---
                for s in range(N_SUP):
                    logits, _, y_new_emb, z_new = model([x_emb, y_curr_emb, z_curr])
                    
                    # Calcul de l'erreur brute
                    loss_raw = loss_fn_ce(y_true, logits) # [Batch, Len]
                    
                    # Application du Masque : On annule l'erreur sur le padding
                    loss_masked = loss_raw * mask
                    
                    # Moyenne uniquement sur les vrais tokens (pas les zéros)
                    loss_step = tf.reduce_sum(loss_masked) / (tf.reduce_sum(mask) + 1e-9)
                    
                    batch_loss += loss_step
                    
                    # Réinjection
                    y_curr_emb = tf.stop_gradient(y_new_emb)
                    z_curr = tf.stop_gradient(z_new)
                
                final_loss = batch_loss / N_SUP

            # Backprop
            grads = tape.gradient(final_loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0) 
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            total_loss += float(final_loss)
            step += 1
            
            if step % 20 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {final_loss:.4f}")

        avg_loss = total_loss/step if step > 0 else 0
        print(f"✅ Fin Epoque {epoch+1} - Moyenne Loss: {avg_loss:.4f}")

    # Sauvegarde propre
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    model.save_weights("saved_models/trm_paper_version.weights.h5")
    print("Sauvegarde terminée.")

if __name__ == "__main__":
    train()