import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
import time

# ──────────────────────────────────────────────────────────────
#  ΜΕΡΟΣ 1: K-D TREE
#  Δομή για πολυδιάστατη εύρεσης αριθμητικών χαρακτηριστικών
#  Χρησιμοποιούμε 5 διαστάσεις: budget, revenue, runtime, popularity, vote_average
# ──────────────────────────────────────────────────────────────

class KDNode:
    """
    Αναπαριστά έναν κόμβο στο k-d tree.
    Κάθε κόμβος αποθηκεύει ένα σημείο στον 5-διάστατο χώρο
    μαζί με τα πλήρη δεδομένα της αντίστοιχης ταινίας.
    """
    def __init__(self, point, data, axis):
        self.point = point   # 5D διάνυσμα: [budget, revenue, runtime, popularity, vote_average]
        self.data  = data    # πλήρης εγγραφή ταινίας (dictionary με όλα τα πεδία)
        self.axis  = axis    # η διάσταση που χρησιμοποιείται για διαχωρισμό (0-4)
        self.left  = None    # αριστερό παιδί: ταινίες με τιμή < τρέχοντος κόμβου
        self.right = None    # δεξί παιδί:     ταινίες με τιμή ≥ τρέχοντος κόμβου


class KDTree:
    """
    Δένδρο k διαστάσεων για αποδοτική αναζήτηση εύρους και k-NN.
    Εναλλάσσει διαστάσεις σε κάθε επίπεδο για ισορροπημένη δομή.
    """
    def __init__(self):
        self.root = None  # ρίζα του δένδρου, αρχικά κενή

    # ── Κατασκευή Δένδρου ──────────────────────────────────────

    def build(self, points, records, depth=0):
        """
        Αναδρομική κατασκευή του k-d tree.
        Σε κάθε επίπεδο επιλέγει την επόμενη διάσταση (axis),
        βρίσκει τη διάμεση τιμή και χωρίζει τα δεδομένα στα δύο.
        """
        # Βασική περίπτωση: αν δεν υπάρχουν σημεία, επιστρέφουμε None
        if not len(points):
            return None

        # Επιλογή διάστασης: εναλλαγή 0→1→2→3→4→0→1→... ανάλογα με το βάθος
        axis = depth % points.shape[1]

        # Ταξινόμηση σημείων ως προς την τρέχουσα διάσταση
        # np.argsort επιστρέφει τους δείκτες που θα ταξινομούσαν τον πίνακα
        idx = np.argsort(points[:, axis])

        # Εύρεση διάμεσης θέσης (μέσο του ταξινομημένου πίνακα)
        mid = len(idx) // 2

        # Δημιουργία κόμβου με το διάμεσο σημείο
        node = KDNode(points[idx[mid]], records[idx[mid]], axis)

        # Αναδρομή: κατασκευή αριστερού υποδένδρου (σημεία πριν τη διάμεση)
        node.left  = self.build(points[idx[:mid]],    [records[i] for i in idx[:mid]],    depth+1)

        # Αναδρομή: κατασκευή δεξιού υποδένδρου (σημεία μετά τη διάμεση)
        node.right = self.build(points[idx[mid+1:]], [records[i] for i in idx[mid+1:]], depth+1)

        return node

    def fit(self, points, records):
        """
        Εκκίνηση κατασκευής δένδρου από τα δεδομένα.
        Καλείται μία φορά κατά τη φόρτωση.
        """
        self.root = self.build(np.array(points), records)

    # ── Ερώτημα Εύρους (Range Query) ───────────────────────────

    def range_query(self, lo, hi, node=None, first=True):
        """
        Επιστρέφει όλες τις ταινίες που βρίσκονται εντός ορίων:
        lo[i] ≤ σημείο[i] ≤ hi[i] για κάθε διάσταση i.

        Χρησιμοποιεί pruning: αγνοεί κλάδους που σίγουρα δεν περιέχουν έγκυρα αποτελέσματα.
        """
        # Αν είναι η πρώτη κλήση, ξεκινάμε από τη ρίζα
        node = self.root if first else node

        # Βασική περίπτωση: κενός κόμβος
        if node is None:
            return []

        results = []

        # Έλεγχος αν το τρέχον σημείο ικανοποιεί ΟΛΕΣ τις συνθήκες
        if all(lo[i] <= node.point[i] <= hi[i] for i in range(len(lo))):
            results.append(node.data)  # προσθήκη στα αποτελέσματα

        ax = node.axis  # τρέχουσα διάσταση διαχωρισμού

        # Pruning: ψάξε αριστερά ΜΟΝΟ αν το κάτω όριο είναι ≤ τιμή κόμβου
        # (διαφορετικά όλα τα αριστερά σημεία είναι εκτός ορίων)
        if lo[ax] <= node.point[ax]:
            results += self.range_query(lo, hi, node.left,  False)

        # Pruning: ψάξε δεξιά ΜΟΝΟ αν το άνω όριο είναι ≥ τιμή κόμβου
        # (διαφορετικά όλα τα δεξιά σημεία είναι εκτός ορίων)
        if hi[ax] >= node.point[ax]:
            results += self.range_query(lo, hi, node.right, False)

        return results

    # ── K Πλησιέστεροι Γείτονες (k-NN) ────────────────────────

    def knn(self, query, k, node=None, best=None, first=True):
        """
        Βρίσκει τις k ταινίες με τη μικρότερη Ευκλείδεια απόσταση
        από το σημείο ερωτήματος.

        Ψάχνει πάντα το κοντινό υποδένδρο πρώτα και το μακρινό
        μόνο αν χρειάζεται (έξυπνο pruning).
        """
        # Αρχικοποίηση για την πρώτη κλήση
        node = self.root if first else node
        best = []          if first else best

        # Βασική περίπτωση: κενός κόμβος
        if node is None:
            return best

        # Υπολογισμός Ευκλείδειας απόστασης: root of Σ(point - query)^2
        dist = np.linalg.norm(node.point - query)

        # Ενημέρωση λίστας k καλύτερων αποτελεσμάτων
        if len(best) < k or dist < best[-1][0]:
            best.append((dist, node.data))
            best.sort(key=lambda x: x[0])  # ταξινόμηση κατά αύξουσα απόσταση
            if len(best) > k:
                best.pop()  # αφαίρεση χειρότερου αν υπερβούμε k

        ax = node.axis  # τρέχουσα διάσταση διαχωρισμού

        # Επιλογή κοντινού και μακρινού υποδένδρου βάσει διάστασης
        near, far = (node.left, node.right) if query[ax] < node.point[ax] else (node.right, node.left)

        # Αναζήτηση κοντινού υποδένδρου πρώτα (πιο πιθανό να έχει καλά αποτελέσματα)
        self.knn(query, k, near, best, False)

        # Αναζήτηση μακρινού υποδένδρου ΜΟΝΟ αν δεν έχουμε βρει k αποτελέσματα ακόμα, Ή η απόσταση στη διάσταση διαχωρισμού < k-οστή καλύτερη απόσταση
        if len(best) < k or abs(query[ax] - node.point[ax]) < best[-1][0]:
            self.knn(query, k, far, best, False)

        return best


# ──────────────────────────────────────────────────────────────
#  ΜΕΡΟΣ 2: LSH (Locality Sensitive Hashing)
#  Για αποδοτική αναζήτηση κειμενικής ομοιότητας στα genres
#  Χρησιμοποιεί MinHash + Banding για γρήγορη εύρεση παρόμοιων ταινιών
# ──────────────────────────────────────────────────────────────

class LSH:
    """
    Locality Sensitive Hashing με MinHash signatures και Banding.

    Αντί να συγκρίνει με 946K ταινίες, βρίσκει υποψήφιες μέσω
    hash tables και συγκρίνει μόνο με αυτά (~100-1000 ταινίες).
    """
    def __init__(self, n_hashes=100, n_bands=20):
        self.n_hashes   = n_hashes          # αριθμός hash functions για MinHash
        self.n_bands    = n_bands            # αριθμός bands για banding technique
        self.rows       = n_hashes // n_bands  # στοιχεία ανά band (100/20 = 5)
        self.tables     = [defaultdict(list) for _ in range(n_bands)]  # 20 hash tables
        self.signatures = {}                 # αποθήκη: {movie_id: [100 αριθμοί]}

    # ── Shingling ───────────────────────────────────────────────

    def _shingle(self, text, k=3):
        """
        Μετατρέπει κείμενο σε σύνολο character 3-grams (shingles).
        Παράδειγμα: "Action" → {"act", "cti", "tio", "ion"}

        Τα shingles επιτρέπουν ανίχνευση ομοιότητας ακόμα και με
        μικρές παραλλαγές στο κείμενο.
        """
        t = text.lower().strip()

        # Αν το κείμενο είναι πολύ κοντό (π.χ. κενό genre), επιστρέφουμε
        # ένα fallback shingle για να αποφύγουμε crash στο min()
        if len(t) < k:
            return {"__empty__"}

        # Δημιουργία όλων των k-grams: "act", "cti", "tio", ...
        return set(t[i:i+k] for i in range(len(t)-k+1))

    # ── MinHash Signature ───────────────────────────────────────

    def _signature(self, shingles):
        """
        Δημιουργεί MinHash signature: λίστα 100 αριθμών που αναπαριστά
        το κείμενο συμπαγώς. 
        """
        # Επιπλέον ασφάλεια αν shingles είναι κενό
        if not shingles:
            shingles = {"__empty__"}

        # Για κάθε μία από τις 100 hash functions, βρες το ελάχιστο hash
        return [min(int(hashlib.md5(f"{s}_{i}".encode()).hexdigest(), 16)
                    for s in shingles)
                for i in range(self.n_hashes)]

    # ── εύρεση (Banding) ───────────────────────────────────

    def add(self, uid, text):
        """
        Ευρετηριάζει μια ταινία στα hash tables.

        Banding Technique:
        - Χωρίζει τη signature (100 στοιχεία) σε 20 bands των 5
        - Κάνει hash κάθε band και τοποθετεί τη ταινία στο bucket
        - Παρόμοιες ταινίες θα ταιριάξουν σε τουλάχιστον ένα band
        """
        # δημιουργία signature για το κείμενο
        sig = self._signature(self._shingle(text))

        # αποθήκευση signature για μελλοντική σύγκριση
        self.signatures[uid] = sig

        # banding - τοποθέτηση σε 20 hash tables
        for b in range(self.n_bands):
            # Εξαγωγή των 5 στοιχείων του τρέχοντος band
            band_slice = sig[b*self.rows : (b+1)*self.rows]

            # Hash του band → bucket ID
            bucket = hash(tuple(band_slice))

            # Τοποθέτηση της ταινίας στο αντίστοιχο bucket
            self.tables[b][bucket].append(uid)

    # ── Ερώτημα Ομοιότητας ─────────────────────────────────────

    def query(self, text, top_k=10):
        """
        Βρίσκει τις top_k πιο παρόμοιες ταινίες για δεδομένο κείμενο.
        Δημιουργία signature για το query
        Εύρεση candidates (ταινίες που μοιράζονται τουλάχιστον 1 bucket)
        Υπολογισμός similarity ΜΟΝΟ για candidates
        Επιστροφή top-k
        """
        # signature για το κείμενο ερωτήματος
        sig   = self._signature(self._shingle(text))

        # εύρεση candidates μέσω banding
        # Ταινίες που βρίσκονται στο ίδιο bucket έστω και για ΕΝΑ band
        cands = set()
        for b in range(self.n_bands):
            bucket = hash(tuple(sig[b*self.rows:(b+1)*self.rows]))
            cands.update(self.tables[b][bucket])  # προσθήκη στα candidates

        # υπολογισμός Jaccard similarity για κάθε candidate
        # similarity = (κοινά στοιχεία signatures) / n_hashes
        scored = sorted(
            ((uid, sum(sig[i] == self.signatures[uid][i]
                       for i in range(self.n_hashes)) / self.n_hashes)
             for uid in cands),
            key=lambda x: x[1], reverse=True  # ταξινόμηση κατά φθίνουσα similarity
        )

        # επιστροφή top-k αποτελεσμάτων
        return scored[:top_k]


# ──────────────────────────────────────────────────────────────
#  ΚΑΘΟΛΙΚΕΣ ΣΤΑΘΕΡΕΣ
# ──────────────────────────────────────────────────────────────

NUMERIC = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']  # 5 αριθμητικά για k-d tree
TEXT    = 'genre_names'   # κειμενικό χαρακτηριστικό για LSH


# ──────────────────────────────────────────────────────────────
#  ΜΕΡΟΣ 3: ΕΝΟΠΟΙΗΜΕΝΟ ΣΥΣΤΗΜΑ
#  Συνδυάζει k-d tree + LSH για διφασική αναζήτηση
# ──────────────────────────────────────────────────────────────

class MovieIndex:
    """
    Ενοποιημένο σύστημα εύρεσης ταινιών.
    Φάση 1: k-d tree για αριθμητικά ερωτήματα εύρους
    Φάση 2: LSH για κειμενική ομοιότητα
    """
    def __init__(self):
        self.kd  = KDTree()  # δομή για αριθμητική αναζήτηση
        self.lsh = LSH()     # δομή για κειμενική ομοιότητα
        self.df  = None      # αποθήκευση του dataframe για ανάκτηση δεδομένων

    # ── Φόρτωση και εύρεση ────────────────────────────────

    def load(self, filepath, sample=None):
        """
        Φορτώνει το dataset και χτίζει και τις δύο δομές εύρεσης.
        """
        print("Loading data...")

        # Φόρτωση  αρχείου


        df = pd.read_csv(filepath)

        # Προαιρετικό sampling για γρήγορα tests
        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)

        # Καθαρισμός αριθμητικών: μετατροπή σε float, NaN → 0
        for col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Καθαρισμός κειμένου: κενά → empty string
        df[TEXT] = df[TEXT].fillna('').astype(str)
        self.df  = df

        # Κατασκευή k-d tree ──
        t = time.time()
        records = df.to_dict('records')          # μετατροπή κάθε γραμμής σε dictionary
        self.kd.fit(df[NUMERIC].values, records) # fit με τις 5 αριθμητικές στήλες
        print(f"k-d tree built in {time.time()-t:.2f}s")

        #Ενημέρωση προόδου για την δομή LSH
        # Build LSH
        t = time.time()
        for _, row in df.iterrows():
            self.lsh.add(row.get('id', _), row[TEXT])
        print(f"LSH built in {time.time()-t:.2f}s")


    # ── Range Query (μόνο k-d tree) ────────────────────────────

    def range_query(self, lo={}, hi={}):
        """
        Βρίσκει ταινίες εντός αριθμητικών ορίων.
        lo: κάτω όρια (π.χ. {'budget': 10_000_000})
        hi: άνω όρια  (π.χ. {'budget': 50_000_000})
        Αν παραλειφθεί χαρακτηριστικό, χρησιμοποιεί -inf/+inf
        """
        # Μετατροπή dictionary σε numpy array (με -inf/+inf για παραλειπόμενα)
        lo_arr = np.array([lo.get(f, -np.inf) for f in NUMERIC])
        hi_arr = np.array([hi.get(f,  np.inf) for f in NUMERIC])

        t   = time.time()
        res = self.kd.range_query(lo_arr, hi_arr)
        print(f"Range query → {len(res)} results in {time.time()-t:.4f}s")
        return res

    # ── Similarity Query (μόνο LSH) ────────────────────────────

    def sim_query(self, text, top_k=10):
        """
        Βρίσκει ταινίες με παρόμοια genres βάσει κειμενικής ομοιότητας.
        text: κείμενο genres προς αναζήτηση (π.χ. "Action Adventure")
        top_k: πόσα αποτελέσματα να επιστρέψει
        """
        t    = time.time()
        hits = self.lsh.query(text, top_k)  # (movie_id, similarity_score) pairs

        # Ανάκτηση πλήρων δεδομένων για κάθε αποτέλεσμα
        results = []
        for uid, score in hits:
            row = self.df[self.df['id'] == uid]
            if not row.empty:
                m = row.iloc[0].to_dict()
                m['similarity'] = score  # προσθήκη similarity score
                results.append(m)

        print(f"Similarity query → {len(results)} results in {time.time()-t:.4f}s")
        return results

    # ── Combined Query (k-d tree + LSH) ────────────────────────

    def combined_query(self, lo={}, hi={}, text='', top_k=10):
        """
        Διφασική αναζήτηση
        Φάση 1: k-d tree φιλτράρει βάσει αριθμητικών 
        Φάση 2: LSH βρίσκει τα πιο παρόμοια μεταξύ των φιλτραρισμένων
        """
        # ── Φάση 1: Range filtering με k-d tree ──
        filtered = self.range_query(lo, hi)
        print(f"After range filter: {len(filtered)} movies")

        # ── Φάση 2: Similarity ranking με LSH ──
        # Χτίζουμε ΠΡΟΣΩΡΙΝΟ LSH index μόνο για τις φιλτραρισμένες ταινίες
        # (πολύ πιο γρήγορο από το να ψάξουμε σε όλες)
        temp = LSH()
        for m in filtered:
            temp.add(m.get('id', m['title']), m.get(TEXT, ''))

        t    = time.time()
        hits = temp.query(text, top_k)  # αναζήτηση στο μικρό σύνολο

        # Δημιουργία lookup map για γρήγορη ανάκτηση δεδομένων
        id_map = {m.get('id', m['title']): m for m in filtered}

        results = []
        for uid, score in hits:
            if uid in id_map:
                id_map[uid]['similarity'] = score  # προσθήκη similarity score
                results.append(id_map[uid])

        print(f"After LSH filter:  {len(results)} movies in {time.time()-t:.4f}s")
        return results


# ──────────────────────────────────────────────────────────────
#  ΜΕΡΟΣ 3: INTERACTIVE QUERY
#  εδώ γίνονται διαδραστικά queries 
# ──────────────────────────────────────────────────────────────

def get_range_input(feature_name):
    """
    Ζητά από τον χρήστη κάτω και άνω όριο για ένα χαρακτηριστικό.
    Αν πατήσει Enter χωρίς τιμή, το παραλείπει (χρησιμοποιεί -inf/+inf).
    Επιστρέφει (lo_value, hi_value) ή None αν παραλειφθεί.
    """
    print(f"\n  [{feature_name}]")

    # Κάτω όριο
    lo_input = input(f"    Κάτω όριο (Enter για παράλειψη): ").strip()

    # Άνω όριο
    hi_input = input(f"    Άνω όριο  (Enter για παράλειψη): ").strip()

    # Αν και τα δύο είναι κενά, παραλείπουμε το χαρακτηριστικό
    if not lo_input and not hi_input:
        print(f"    → Παραλείφθηκε")
        return None

    # Μετατροπή σε float (αν δόθηκε τιμή)
    lo_val = float(lo_input) if lo_input else None
    hi_val = float(hi_input) if hi_input else None

    print(f"    → Εύρος: {lo_val if lo_val is not None else '-∞'}  έως  {hi_val if hi_val is not None else '+∞'}")
    return (lo_val, hi_val)


def run_interactive_query(db):
    """
    Κύρια συνάρτηση διαδραστικής αναζήτησης.
    Ρωτά τον χρήστη για κάθε χαρακτηριστικό και εκτελεί combined query.
    """
    print("\n" + "="*55)
    print("  ΔΙΑΔΡΑΣΤΙΚΗ ΑΝΑΖΗΤΗΣΗ ΤΑΙΝΙΩΝ")
    print("  Πατήστε Enter για παράλειψη οποιουδήποτε ορίου")
    print("="*55)

    lo = {}  # dictionary με κάτω όρια
    hi = {}  # dictionary με άνω όρια

    # ── Ρώτα για κάθε αριθμητικό χαρακτηριστικό ──
    print("\n── Αριθμητικά Όρια ──")

    for feature in NUMERIC:
        result = get_range_input(feature)

        # Αν ο χρήστης έδωσε τιμές, αποθήκευσε τα όρια
        if result is not None:
            lo_val, hi_val = result
            if lo_val is not None: lo[feature] = lo_val  # αποθήκευση κάτω ορίου
            if hi_val is not None: hi[feature] = hi_val  # αποθήκευση άνω ορίου

    # ── Ρώτα για κείμενο LSH ──
    print("\n── Κειμενική Ομοιότητα (LSH) ──")
    text_input = input("  Genres προς αναζήτηση (π.χ. 'Action Drama')\n  → ").strip()

    # ── Ρώτα πόσα αποτελέσματα θέλει ──
    print("\n── Αποτελέσματα ──")
    top_k_input = input("  Πόσα αποτελέσματα θέλεις; (Enter για 10): ").strip()
    top_k = int(top_k_input) if top_k_input.isdigit() else 10

    # ── Εμφάνιση σύνοψης ερωτήματος ──
    print("\n" + "-"*55)
    print("  ΠΑΡΑΜΕΤΡΟΙ ΑΝΑΖΗΤΗΣΗΣ:")
    if lo or hi:
        for feature in NUMERIC:
            if feature in lo or feature in hi:
                lo_str = f"{lo[feature]:,.0f}" if feature in lo else "-∞"
                hi_str = f"{hi[feature]:,.0f}" if feature in hi else "+∞"
                print(f"    {feature}: {lo_str} → {hi_str}")
    else:
        print("    (Δεν δόθηκαν αριθμητικά όρια)")

    print(f"    genres: '{text_input}'" if text_input else "    (Δεν δόθηκε κείμενο)")
    print(f"    top_k:  {top_k}")
    print("-"*55)

    # ── Εκτέλεση κατάλληλου query ──
    if (lo or hi) and text_input:
        # Και αριθμητικά ΚΑΙ κείμενο → combined query
        print("\n  Εκτέλεση Combined Query (k-d tree + LSH)...")
        results = db.combined_query(lo=lo, hi=hi, text=text_input, top_k=top_k)

    elif (lo or hi):
        # Μόνο αριθμητικά → range query
        print("\n  Εκτέλεση Range Query (μόνο k-d tree)...")
        results = db.range_query(lo=lo, hi=hi)
        results = results[:top_k]  # κράτα μόνο top_k

    elif text_input:
        # Μόνο κείμενο → similarity query
        print("\n  Εκτέλεση Similarity Query (μόνο LSH)...")
        results = db.sim_query(text=text_input, top_k=top_k)

    else:
        # Τίποτα → δεν μπορούμε να κάνουμε query
        print("\n  ⚠ Δεν δόθηκε κανένα κριτήριο αναζήτησης!")
        return

    # ── Εμφάνιση αποτελεσμάτων ──
    print(f"\n  ΑΠΟΤΕΛΕΣΜΑΤΑ ({len(results)} ταινίες):")
    print("="*55)

    if not results:
        print("  Δεν βρέθηκαν ταινίες με αυτά τα κριτήρια.")
    else:
        for i, m in enumerate(results, 1):
            print(f"\n  {i}. {m.get('title', 'Άγνωστη')}")
            print(f"     Budget:  ${m.get('budget', 0):>15,.0f}")
            print(f"     Revenue: ${m.get('revenue', 0):>15,.0f}")
            print(f"     Runtime: {m.get('runtime', 0):.0f} λεπτά")
            print(f"     Rating:  {m.get('vote_average', 0):.1f}/10")
            print(f"     Genres:  {m.get('genre_names', '-')}")
            if 'similarity' in m:
                print(f"     Similarity: {m['similarity']:.2f}")

    print("\n" + "="*55)


# ──────────────────────────────────────────────────────────────
#  ΕΚΚΙΝΗΣΗ
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Φόρτωση δεδομένων
    db = MovieIndex()

    db.load('data_movies_clean.csv', sample=50000)

    # Εκτέλεση διαδραστικής αναζήτησης
    run_interactive_query(db)









'''
Debug code used during development, please ignore us :)


#Ενημέρωση προόδου για την δομή LSH
t = time.time()
print("Building LSH index (this may take a while)...")
for _, row in df.iterrows():
    self.lsh.add(row.get('id', _), row[TEXT])  
    print(f"LSH built in {time.time()-t:.2f}s")    


#Για αλλαγή ανάγνοση απο csv σε xlsx -> άλλαξε και στην μέθοδο load της κλάσης movieindex γραμμή ~304 
#db.load('data_movies_clean.xlsx', sample=50000)


#Για αλλαγή ανάγνοση απο csv σε xlsx -> άλλαξε και στην main στο τέλος του κώδικα
#df = pd.read_excel(filepath)

'''    