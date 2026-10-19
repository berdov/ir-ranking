"""Independent brute-force oracle and input/CLI regression tests."""

from array import array
import csv
import hashlib
import inspect
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solution


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_tsv(path, rows):
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def words(text):
    # No calls to the production tokenizer or index: an independent oracle.
    return {word.lower() for word in re.findall(r"\w+", text)}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.data = self.root / "данные с пробелом"
        self.data.mkdir()
        self.index = self.root / "index"
        self.output = self.root / "submission.csv"
        self.docs = [
            ("001", "https://urlonly.example", '"КОШКА"', "спит. Кошка!"),
            ("1", "https://кошка.рф", "Собака", "БЕЖИТ"),
            ("7000000000000", "https://x", "Ёлка 2026", "кот_дом A-b"),
            ("doc-x", "https://x", "кошка", "бежит"),
            ("empty", "https://urlonly", "", ""),
            ("кириллица", "", "Елка", "не ёлка"),
            ("comma,id", "", "alpha", "beta\tgamma"),
        ]
        self.queries = [
            ("q01", "кошка"), ("q1", "кошка БЕЖИТ"),
            ("q2", "КОШКА кошка кошка"), ("q3", "urlonly"),
            ("q4", "несуществующее"), ("q5", "ЁЛКА 2026"),
            ("q6", "!!!"), ("q7", ""), ("q8", "кот_дом A-b"),
            ("q9", "кошки"), ("q10", "елка"), ("q11", "alpha beta gamma"),
        ]
        self.objects = [(f"o{i:04}", query[0], doc[0])
                        for i, (query, doc) in enumerate(
                            (q, d) for q in self.queries for d in self.docs)]
        random.Random(42).shuffle(self.objects)
        self.save_inputs()
        solution.build_index(self.data, self.index)

    def save_inputs(self):
        write_tsv(self.data / solution.DOCUMENTS_FILE, self.docs)
        write_tsv(self.data / solution.QUERIES_FILE, self.queries)
        write_csv(self.data / "objects.csv", ["ObjectId", "QueryId", "DocumentId"], self.objects)
        write_csv(self.data / "sample_submission.csv", ["ObjectId", "Label"],
                  [(row[0], i % 2) for i, row in enumerate(reversed(self.objects))])

    def run_submission(self):
        solution.generate_submission(self.data, self.index, self.output)

    def assert_oracle(self):
        document_words = {row[0]: words(row[2] + " " + row[3]) for row in self.docs}
        query_words = {row[0]: words(row[1]) for row in self.queries}
        expected = [[object_id, str(int(query_words[query_id] <= document_words[document_id]))]
                    for object_id, query_id, document_id in self.objects]
        with self.output.open(encoding="utf-8", newline="") as source:
            self.assertEqual(list(csv.reader(source)), [["ObjectId", "Label"], *expected])

    def test_exact_preprocess_from_course_template(self):
        expected = (
            "def preprocess(text):\n"
            '    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:\n'
            '    ей же размечен эталон на Kaggle."""\n'
            "    tokenizer = tokenize.RegexpTokenizer(r'\\w+')\n"
            "    tokens = tokenizer.tokenize(text)\n"
            "    return [token.lower() for token in tokens]\n"
        )
        self.assertEqual(inspect.getsource(solution.preprocess), expected)

    def test_all_pairs_against_independent_oracle_without_corpus(self):
        before = hashlib.sha256((self.index / solution.INDEX_FILE).read_bytes()).hexdigest()
        (self.data / solution.DOCUMENTS_FILE).unlink()
        self.run_submission()
        self.assert_oracle()
        self.assertEqual(before, hashlib.sha256(
            (self.index / solution.INDEX_FILE).read_bytes()).hexdigest())
        first = self.output.read_bytes()
        self.run_submission()
        self.assertEqual(first, self.output.read_bytes())

    def test_random_corpus_against_oracle(self):
        rng = random.Random(2026)
        vocabulary = ["кот", "КОТ", "кошка", "Ёлка", "елка", "A", "b", "42", "токен_1"]
        self.docs = [(f"d{n * 73}", "https://neverindex", " ".join(rng.choices(vocabulary, k=3)),
                      "-".join(rng.choices(vocabulary, k=rng.randrange(20)))) for n in range(150)]
        self.queries = [(f"q{n}", " ".join(rng.choices(vocabulary + ["unknown"],
                                                      k=rng.randrange(6)))) for n in range(60)]
        self.objects = [(str(i), q[0], d[0]) for i, (q, d) in enumerate(
            (q, d) for q in self.queries for d in self.docs)]
        rng.shuffle(self.objects)
        self.save_inputs()
        solution.build_index(self.data, self.index)
        self.run_submission()
        self.assert_oracle()

    def test_both_required_cli_modes(self):
        script = str(Path(solution.__file__).resolve())
        for args in (["--build_index"], [f"--submission_file={self.output}"]):
            run = subprocess.run([sys.executable, script, *args,
                                  f"--index_dir={self.index}", str(self.data)],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assert_oracle()

    def test_bad_inputs_preserve_previous_submission(self):
        for kind in ("query", "document", "object_duplicate", "sample_duplicate", "sample_missing",
                     "sample_extra", "query_duplicate", "header", "truncated"):
            with self.subTest(kind=kind):
                self.save_inputs()
                sentinel = "previous successful submission\n"
                self.output.write_text(sentinel)
                if kind in ("query", "document"):
                    rows = list(self.objects)
                    rows[0] = (rows[0][0], "missing" if kind == "query" else rows[0][1],
                               "missing" if kind == "document" else rows[0][2])
                    write_csv(self.data / "objects.csv", ["ObjectId", "QueryId", "DocumentId"], rows)
                elif kind == "object_duplicate":
                    with (self.data / "objects.csv").open("a") as target:
                        csv.writer(target).writerow(self.objects[0])
                elif kind == "sample_duplicate":
                    with (self.data / "sample_submission.csv").open("a") as target:
                        csv.writer(target).writerow((self.objects[0][0], "0"))
                elif kind in ("sample_missing", "sample_extra"):
                    rows = [(row[0], "0") for row in self.objects]
                    rows = rows[1:] if kind == "sample_missing" else rows + [("extra", "1")]
                    write_csv(self.data / "sample_submission.csv", ["ObjectId", "Label"], rows)
                elif kind == "query_duplicate":
                    write_tsv(self.data / solution.QUERIES_FILE, self.queries + [self.queries[0]])
                elif kind == "header":
                    (self.data / "objects.csv").write_text("bad,header\n")
                else:
                    (self.data / "objects.csv").write_text("ObjectId,QueryId,DocumentId\nx,q\n")
                with self.assertRaises(ValueError):
                    self.run_submission()
                self.assertEqual(self.output.read_text(), sentinel)
                self.assertEqual(list(self.root.glob(".submission-*")), [])

    def test_failed_rebuild_keeps_old_index(self):
        old = (self.index / solution.INDEX_FILE).read_bytes()
        write_tsv(self.data / solution.DOCUMENTS_FILE, self.docs + [self.docs[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate DocumentId"):
            solution.build_index(self.data, self.index)
        self.assertEqual(old, (self.index / solution.INDEX_FILE).read_bytes())

    def test_output_cannot_replace_inputs(self):
        for path in [self.data / "objects.csv", self.data / solution.DOCUMENTS_FILE,
                     self.index / solution.INDEX_FILE]:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "overwrite"):
                solution.generate_submission(self.data, self.index, path)

    def test_missing_index(self):
        with self.assertRaisesRegex(ValueError, "Index not found"):
            solution.BooleanIndex(self.root / "missing")

    def test_empty_corpus(self):
        self.docs, self.objects = [], []
        self.save_inputs()
        solution.build_index(self.data, self.index)
        self.run_submission()
        self.assertEqual(self.output.read_text(), "ObjectId,Label\n")


class PostingTests(unittest.TestCase):
    def test_roundtrip_and_varint_boundaries(self):
        rng = random.Random(11)
        for values in ([], [0], [127, 128, 255, 256, 16384, 0xFFFFFFFF],
                       sorted(rng.sample(range(1_000_000), 20_000))):
            with self.subTest(length=len(values)):
                self.assertEqual(list(solution.decode_postings(
                    solution.encode_postings(array("I", values)))), values)

    def test_corrupt_varint(self):
        with self.assertRaises(ValueError):
            list(solution.decode_postings(b"\x80"))
        with self.assertRaises(ValueError):
            list(solution.decode_postings(b"\x80" * 6))


if __name__ == "__main__":
    unittest.main()
