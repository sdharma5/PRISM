PRISM static demo
=================

Self-contained. No backend required.

Run it:

    cd this-folder
    python3 -m http.server 8080

then open http://localhost:8080/overview/

Opening the .html files directly with file:// will not work — the app loads
JavaScript by absolute path, which browsers block on the file:// protocol.
Any static server works; python3 is just the one everyone already has.

Data: the three generated demo patients (Sarah, Priya, Maya). These are real
model outputs, produced by running the actual inference pipeline, not
hand-written values.

Not included: /intake and /inputs/voice (need the live API), /care and
/recommendations (need server routes).
