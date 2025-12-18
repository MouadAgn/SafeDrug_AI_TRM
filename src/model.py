import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TinyBlock(layers.Layer):
    """Un bloc simple : Norm -> MLP -> Residual"""
    def __init__(self, d):
        super().__init__()
        self.ln = layers.LayerNormalization()
        self.fc1 = layers.Dense(4 * d, activation="swish") 
        self.fc2 = layers.Dense(d)

    def call(self, u):
        h = self.ln(u)
        h = self.fc1(h)
        h = self.fc2(h)
        return u + h

class TRM_Official(keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, d=64, max_len=128, n=6, T=3):
        super().__init__()
        self.d = d
        self.n = n 
        self.T = T 
        
        # Embeddings
        self.emb_x = layers.Embedding(input_vocab_size, d)
        self.emb_y = layers.Embedding(output_vocab_size, d)
        
        # Positional Encoding
        self.pos_emb = self.add_weight(name="pos", shape=(1, max_len, d), initializer="random_normal", trainable=True)

        # Le Tiny Network (2 couches selon le papier [cite: 8])
        self.block1 = TinyBlock(d)
        self.block2 = TinyBlock(d)

        # Tête de sortie unique (On a supprimé halt_head pour éviter le warning)
        self.to_vocab = layers.Dense(output_vocab_size)

    def net(self, u):
        u = self.block1(u)
        u = self.block2(u)
        return u

    def latent_recursion(self, x, y, z):
        # 1. Mise à jour du raisonnement (z) n fois
        for _ in range(self.n):
            u_z = x + y + z 
            z = self.net(u_z) 
            
        # 2. Mise à jour de la réponse (y) une fois
        u_y = y + z
        y_out = self.net(u_y)
        
        return y_out, z

    def call(self, inputs):
        # inputs: [x_emb, y_emb, z_features]
        x, y, z = inputs
        
        # Boucle de Récursion Externe (T)
        for i in range(self.T):
            if i < self.T - 1:
                # Stop gradient pour les premières passes (économise mémoire)
                y, z = tf.stop_gradient(self.latent_recursion(x, y, z)[0]), tf.stop_gradient(self.latent_recursion(x, y, z)[1])
            else:
                # Gradient actif pour la dernière passe
                y, z = self.latent_recursion(x, y, z)
                
        logits = self.to_vocab(y)
        # On retourne None pour halt_logits pour l'instant
        return logits, None, y, z