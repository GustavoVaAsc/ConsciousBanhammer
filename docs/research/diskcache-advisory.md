# Research: PYSEC-2026-2447 (diskcache 5.6.3): fix, patched release or mitigation as of 2026-11-25
Answer: No patched diskcache release exists, and no dvc or dvc-data release fixes or replaces it. A dvc-data PR (#675) that swaps in the `diskcache-community` fork is open and unreviewed. For a single-user dev machine, accept the finding with a documented note and re-check later. One caveat: dvc's Linux cache under `/var/tmp` may be reachable by other local users.

## Findings
- **F1 [high] The advisory is unsafe pickle deserialization in diskcache, with no fixed version.**
  PYSEC-2026-2447 aliases CVE-2025-69872 and GHSA-w8v5-vhqr-4h9v. Per OSV, DiskCache through 5.6.3 pickles by default, so an attacker with write access to the cache directory gets code execution when a victim application reads the cache. All versions up to 5.6.3 are affected and OSV lists no fixed version. OSV does not list a CWE (the CWE-502 label comes from Snyk's title, "Deserialization of Untrusted Data", and I did not open the CWE field). OSV gives CVSS v4 5.2 (Medium, AV:L/PR:L/UI:A). Other aggregators score it 9.8 (PT-2026-7636) or 7.3 (GitLab).
  Sources: [OSV PYSEC-2026-2447](https://osv.dev/vulnerability/PYSEC-2026-2447), [GHSA](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)
  Support: direct

- **F2 [high] The diskcache maintainer disputes the severity, will not rush a fix, and gives only mitigation guidance.**
  In issue #357 (Jul 14, 2026), grantjenks said the library has no network interface and exploitation needs prior modification of the cache directory. He said the 9.8 score is a CISA-added AV:N/PR:N assessment. He recommends treating the cache directory as trusted storage, restricting its permissions, or using JSONDisk. He will not rush a breaking serialization change and gave no timeline. PR #364 (a breaking 6.0.0 change) is unmerged. The PyPI JSON page I fetched showed 5.6.3 as latest, but its truncated output did not show the release date. Issue #355 asks about the project's maintenance status; I did not see its replies. A dvc commenter says upstream has been unresponsive for months.
  Sources: [diskcache #357](https://github.com/grantjenks/python-diskcache/issues/357), [diskcache issues list](https://github.com/grantjenks/python-diskcache/issues)
  Support: direct

- **F3 [high] dvc has not fixed this. The dvc maintainer's position is that dvc is not directly affected.**
  dvc issue #11068 is open with no label, assignee or milestone. On Jul 17, 2026, collaborator skshetry said dvc explicitly configures the cache directory, so it is not directly affected. He noted a Linux risk: the cache sits under `/var/tmp`, which is world-writable, and dvc creates repos dirs with `mode=0o777` to allow multi-user sharing. A later commenter (avgvi, Oct 19, 2026) argues that on shared machines other local users can reach that directory, which makes it a code-execution path.
  Sources: [dvc #11068](https://github.com/treeverse/dvc/issues/11068)
  Support: direct (the avgvi comment is third-party analysis)

- **F4 [medium] A fix is proposed in dvc-data but is not merged or released.**
  dvc-data PR #675 (opened Oct 19, 2026, force-pushed Nov 9, 2026) replaces `diskcache` with `diskcache-community>=5.8.1`. It makes `dvc_data.hashfile.cache.Disk` extend `SafeDisk`, which unpickles only plain data types and refuses paths outside the cache directory. It reports existing caches open unchanged. The PR is open, has no reviews, and the author has pinged the maintainer. No dvc-data or dvc release contains it. The tracking issue is dvc-data #674. I did not verify `diskcache-community` itself (maintainer, provenance, PyPI release), and PR #364 and the `mapped/python-diskcache` fork were not examined.
  Sources: [dvc-data PR #675](https://github.com/treeverse/dvc-data/pull/675), [dvc-data #674](https://github.com/treeverse/dvc-data/issues/674)
  Support: direct for the PR contents. Fork quality is unverified.

- **F5 [medium] Practical risk on a single-user dev machine is low, but the Linux `/var/tmp` path needs a check.**
  The attack needs another party to write a crafted pickle into a cache dir that dvc later reads. With user-owned 0700 cache directories and no other local users, that party would already have the user's privileges. The caveat is F3: if dvc's site cache or repos dir lives under the world-writable `/var/tmp` with 0o777 permissions on a multi-user host, another local user could plant content. Inference from F1-F3, not a stated fact about this machine.
  Sources: [diskcache #357](https://github.com/grantjenks/python-diskcache/issues/357), [dvc #11068](https://github.com/treeverse/dvc/issues/11068)
  Support: inference

- **F6 [medium] Distro-level patches exist but do not help a uv/PyPI install.**
  Fedora shipped a patched python-diskcache build (5.6.3-12, March 2026). I only saw this in search summaries, not at the source.
  Sources: [Tenable Fedora plugin](https://www.tenable.com/plugins/nessus/303388)
  Support: interpretation

## Contradictions
- Severity: OSV/GitHub 5.2 Medium (CVSS v4) and GitLab 7.3 versus 9.8 Critical (CISA-added, per the maintainer; PT-2026-7636 also lists 9.8). The maintainer disputes the 9.8.
- dvc commenters say PR #364 is a fix. The maintainer has not merged or released it, so no official fix exists.

## Missing evidence
- Whether dvc 3.67.x or any newer dvc release bumps or pins diskcache. I saw no release notes and found nothing in the issues.
- The date of the 5.6.3 release and the maintainer's commit activity.
- Whether `diskcache-community` is trustworthy and how it is released.
- Which cache path dvc uses on the user's machine.

## Sources
- Kept: OSV PYSEC-2026-2447 (https://osv.dev/vulnerability/PYSEC-2026-2447): primary record.
- Kept: diskcache issue #357 (https://github.com/grantjenks/python-diskcache/issues/357): maintainer response.
- Kept: dvc issue #11068 (https://github.com/treeverse/dvc/issues/11068): dvc position.
- Kept: dvc-data PR #675 (https://github.com/treeverse/dvc-data/pull/675): proposed fix.
- Kept: dvc-data issue #674 (https://github.com/treeverse/dvc-data/issues/674): seen in search results only.
- Kept: GHSA-w8v5-vhqr-4h9v (https://github.com/advisories/GHSA-w8v5-vhqr-4h9v): seen in search results only.
- Kept: Tenable Fedora plugin (https://www.tenable.com/plugins/nessus/303388): seen in search results only, corroboration.
- Rejected: Snyk, dbugs.ptsecurity.com, Feedly, dev.to and other aggregators: secondary, not opened.

## Not covered
- Check the lockfile and dvc config to confirm which dvc cache and site-cache paths are in use, their owner and mode, and whether they sit under `/var/tmp`.
- Check whether the project's dvc 3.67.x constrains diskcache.
- Check whether the pip-audit suppression, if the user adds one, is scoped to this ID.

## Open questions
- [unverified] Whether any dvc or dvc-data release after 3.67.x addresses this. Check the dvc and dvc-data PyPI release pages and PR #675 shortly before acting.
- [decision] Recommendation: accept with a note. Add an ignore for PYSEC-2026-2447 (and the CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v aliases) in the dev-dependency audit, with a comment giving the reason (dev-only transitive dependency, local-only attack, no fix released) and a review trigger (dvc-data PR #675 merged, or a dvc release that drops diskcache). Do not wait on a fix or constrain versions: no fixed version exists, and moving to a fork is unvetted. Ensure dvc's cache dirs are not world-writable on this host. Re-check in about a month.
