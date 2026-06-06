### Analisi dell'Data Augmentation Standard (Approccio Iniziale)

Nella fase iniziale del progetto, abbiamo tentato di integrare la Data Augmentation direttamente all'interno della topologia della rete, definendo layer come `RandomFlip` e `RandomRotation`.

#### Comportamento del Codice

Quando definiamo layer di augmentation all'interno di un modello `Sequential` in Keras, queste operazioni diventano parte integrante del grafo computazionale della GPU. Il codice richiamava internamente funzioni di trasformazione geometrica come `ImageProjectiveTransformV3`.

#### Analisi Tecnica

* **Trasformazione Arbitraria:** L'operatore `RandomRotation` non si limita a ruotare l'immagine di 90° (che sarebbe un'operazione di trasposizione semplice), ma applica rotazioni ad angoli arbitrari.
* **Intervento dell'Interpolazione:** Poiché i pixel di un'immagine digitale sono disposti su una griglia discreta, ruotare un'immagine di un angolo non multiplo di 90° richiede di calcolare il nuovo valore di ogni pixel tramite **interpolazione bilineare o bicubica**.
* **Effetto sui Dati:** Per le mappe dei wafer, dove i difetti di tipo `scratch` sono costituiti da linee sottili (spesso larghe un singolo pixel), l'interpolazione agisce come un filtro passa-basso: "spalma" l'intensità del pixel originale su quelli adiacenti. Il risultato è la perdita di nitidezza del graffio, che diventa una traccia sfocata. Il modello, durante l'addestramento, non vede più la caratteristica distintiva del difetto, portando a una drastica riduzione della precisione.

---

### Analisi della "Safe Augmentation" (Approccio Ottimizzato)

Per risolvere i problemi di sfocamento e le incompatibilità hardware (DirectML), abbiamo spostato l'augmentation all'esterno del modello, utilizzando `ImageDataGenerator` di Keras esclusivamente con trasformazioni di riflessione.

#### Comportamento del Codice

Abbiamo rimosso i layer di rotazione dalla funzione `build_optimized_model` e abbiamo configurato il generatore come segue:

```python
datagen_safe = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)
```

#### Analisi Tecnica

* **Assenza di Interpolazione:** Le operazioni `horizontal_flip` e `vertical_flip` sono trasformazioni di **permutazione dei pixel**. In termini di codice, non viene effettuato alcun calcolo matematico tra pixel vicini; le coordinate del pixel `(x, y)` vengono semplicemente rimappate in `(width - x, y)` o `(x, height - y)`.
* **Conservazione della Risoluzione:** Poiché non c'è interpolazione, il graffio mantiene la sua nitidezza originale, lo spessore e il contrasto intatti. La geometria del difetto è preservata al 100%.
* **Efficienza su DirectML:** Spostando queste operazioni all'esterno del modello (gestite dalla CPU in parallelo al training), eliminiamo la necessità di supportare complessi operatori di trasformazione proiettiva sulla GPU, evitando i warning del compilatore e i colli di bottiglia computazionali.
* **Generalizzazione:** Il modello impara che un difetto rimane tale anche se ribaltato (invarianza per riflessione), senza però essere esposto a dati "artificiali" e degradati dalla rotazione.


