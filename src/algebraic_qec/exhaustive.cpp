// Exhaustive search for low-weight nontrivial logical operators of a CSS code.
//
// Usage: exhaustive INPUT [TIME_LIMIT_SECONDS [FIRST_ANCHOR [LAST_ANCHOR]]]
//
// INPUT (written by algebraic_qec.distance.write_input) contains the header
//     n  #checks  #stabilizers  #conjugates  period  max_weight
// followed by one line "count index ..." per row: first the opposite-sector
// check rows C, then the same-sector stabilizer rows S, then a basis L of
// ker(S)/rowspan(C).  The search looks for a word e in ker(C) outside rowspan(S)
// with |e| <= max_weight.  Such a word pairs oddly with some row of L.
//
// Completeness: see the algebraic_qec.distance docstring.  Output: one line per cyclic block
// anchor, then either
//     WITNESS q1 q2 ...                              (exit code 1)
//     PROVED_NO_LOGICAL_WEIGHT_LE W first F last L   (exit code 0)
// or exit code 3 on timeout and 2 on invalid input.

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

constexpr int MAX_QUBITS = 2560;  // capacity of the qubit bitsets
constexpr int MAX_ROWS = 1536;    // capacity of the check/stabilizer bitsets
using Bits = bitset<MAX_QUBITS>;

int n, num_checks, num_stabs, num_conjugates, period, max_weight;
vector<vector<int>> checks, stabs, conjugates;
vector<vector<int>> stabs_of_qubit;  // stabilizer rows containing each qubit
vector<Bits> check_bits, stab_bits;
vector<bitset<MAX_ROWS>> check_column;  // checks touching each qubit
vector<int> overlap_bound;               // floor(|s| / 2) for each stabilizer s
unsigned long long nodes = 0, cycles = 0;
chrono::steady_clock::time_point start;
bool found = false, timed_out = false;
vector<int> witness;
double time_limit = 3600;

struct State {
    Bits excluded;              // qubits that may not be added any more
    bitset<MAX_ROWS> syndrome;  // checks with odd overlap with the support
    array<unsigned char, MAX_ROWS> counts{};  // overlap with each stabilizer
    vector<int> support;
};

double elapsed() { return chrono::duration<double>(chrono::steady_clock::now() - start).count(); }

// Add qubit q to the support.  A minimum-weight logical e overlaps every
// stabilizer s in at most floor(|s|/2) qubits (otherwise e + s is lighter), so
// a violated bound prunes the branch and an attained bound excludes the rest of s.
bool include(State &state, int q) {
    if (state.excluded[q]) return false;
    state.support.push_back(q);
    state.excluded.set(q);
    state.syndrome ^= check_column[q];
    for (int s : stabs_of_qubit[q]) {
        if (++state.counts[s] > overlap_bound[s]) return false;
        if (state.counts[s] == overlap_bound[s]) state.excluded |= stab_bits[s];
    }
    return true;
}

void dfs(State state) {
    if (found || timed_out) return;
    if ((++nodes & 65535) == 0 && elapsed() > time_limit) {
        timed_out = true;
        return;
    }
    if (state.syndrome.none()) {
        // First zero syndrome on this branch: report it if it is nontrivial.
        ++cycles;
        for (const auto &conjugate : conjugates) {
            int parity = 0;
            for (int q : state.support)
                if (find(conjugate.begin(), conjugate.end(), q) != conjugate.end()) parity ^= 1;
            if (parity) {
                found = true;
                witness = state.support;
                return;
            }
        }
        return;
    }
    int remaining = max_weight - static_cast<int>(state.support.size());
    if (remaining <= 0) return;

    // Every unsatisfied check needs at least one more available qubit.
    vector<pair<int, int>> choices;  // (available qubits, check row)
    choices.reserve(80);
    for (int r = 0; r < num_checks; r++) {
        if (!state.syndrome[r]) continue;
        int available = (check_bits[r] & ~state.excluded).count();
        if (available == 0) return;
        choices.emplace_back(available, r);
    }
    sort(choices.begin(), choices.end());

    // Unsatisfied checks with pairwise disjoint available qubits each need a
    // distinct new qubit: a greedy packing gives a lower bound on the weight.
    Bits packed;
    int lower_bound = 0;
    for (auto [available, r] : choices) {
        Bits free_qubits = check_bits[r] & ~state.excluded;
        if ((packed & free_qubits).none()) {
            packed |= free_qubits;
            if (++lower_bound > remaining) return;
        }
    }

    // Branch on the available qubits of the most constrained unsatisfied check,
    // excluding each tried qubit from the later sibling branches.
    int row = choices[0].second;
    vector<pair<int, int>> candidates;  // (-#unsatisfied checks fixed, qubit)
    for (int q : checks[row])
        if (!state.excluded[q])
            candidates.emplace_back(-static_cast<int>((check_column[q] & state.syndrome).count()), q);
    sort(candidates.begin(), candidates.end());
    for (auto [score, q] : candidates) {
        State child = state;
        if (include(child, q)) dfs(child);
        state.excluded.set(q);
        if (found || timed_out) return;
    }
}

bool read_rows(ifstream &in, vector<vector<int>> &rows, int count) {
    rows.resize(count);
    for (auto &row : rows) {
        int weight;
        if (!(in >> weight) || weight < 0 || weight > n) return false;
        row.resize(weight);
        for (int &q : row)
            if (!(in >> q) || q < 0 || q >= n) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " INPUT [TIME_LIMIT [FIRST_ANCHOR [LAST_ANCHOR]]]" << endl;
        return 2;
    }
    ifstream in(argv[1]);
    if (!(in >> n >> num_checks >> num_stabs >> num_conjugates >> period >> max_weight)) return 2;
    if (n > MAX_QUBITS || num_checks > MAX_ROWS || num_stabs > MAX_ROWS) return 2;
    if (period <= 0 || n % period != 0) return 2;
    if (argc > 2) time_limit = stod(argv[2]);
    int first = argc > 3 ? stoi(argv[3]) : 0;
    int last = argc > 4 ? stoi(argv[4]) : n / period;

    if (!read_rows(in, checks, num_checks) || !read_rows(in, stabs, num_stabs) ||
        !read_rows(in, conjugates, num_conjugates))
        return 2;
    check_column.resize(n);
    stabs_of_qubit.resize(n);
    check_bits.resize(num_checks);
    stab_bits.resize(num_stabs);
    overlap_bound.resize(num_stabs);
    for (int r = 0; r < num_checks; r++)
        for (int q : checks[r]) {
            check_column[q].set(r);
            check_bits[r].set(q);
        }
    for (int r = 0; r < num_stabs; r++) {
        overlap_bound[r] = stabs[r].size() / 2;
        for (int q : stabs[r]) {
            stabs_of_qubit[q].push_back(r);
            stab_bits[r].set(q);
        }
    }

    // The checks and stabilizers are invariant under the simultaneous cyclic
    // shift of all blocks of `period` qubits, so some minimum-weight logical
    // has its first occupied block at b with sheet zero at qubit b * period.
    start = chrono::steady_clock::now();
    for (int b = first; b < last; b++) {
        State state;
        for (int q = 0; q < b * period; q++) state.excluded.set(q);
        if (include(state, b * period)) dfs(state);
        cout << "anchor " << b << " nodes " << nodes << " cycles " << cycles << " seconds "
             << elapsed() << " timeout " << timed_out << " found " << found << endl;
        if (found || timed_out) break;
    }
    if (found) {
        cout << "WITNESS";
        for (int q : witness) cout << ' ' << q;
        cout << endl;
        return 1;
    }
    if (timed_out) return 3;
    cout << "PROVED_NO_LOGICAL_WEIGHT_LE " << max_weight << " first " << first << " last " << last
         << endl;
}
