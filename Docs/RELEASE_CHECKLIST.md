# Public release checklist

Before making the repository public:

- confirm the repository/license choice is appropriate for all authors;
- add the final author/citation metadata once the manuscript record is public;
- attach the selected `best.pt` checkpoint to a GitHub Release or archival repository if redistribution is permitted;
- publish SHA-256 checksums for released checkpoints and key result artifacts;
- add small final JSON/CSV result artifacts if journal/conference policy permits;
- verify no raw MRI data, masks, patient identifiers, credentials, or machine-specific absolute paths are present;
- create a clean environment and run the verification stages on the documented dataset layout;
- tag the code version corresponding to the submitted/accepted manuscript.
