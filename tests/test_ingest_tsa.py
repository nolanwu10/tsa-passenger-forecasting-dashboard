import pandas as pd

from tsa_project.ingest_tsa import extract_tsa_rows


def test_extract_tsa_rows_reads_first_html_table():
    html = """
    <html>
      <body>
        <table>
          <thead>
            <tr><th>Date</th><th>Passengers</th></tr>
          </thead>
          <tbody>
            <tr><td>6/3/2026</td><td>2,400,663</td></tr>
            <tr><td>6/2/2026</td><td>2,218,178</td></tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    rows = extract_tsa_rows(html)
    frame = pd.DataFrame(rows, columns=["Date", "Passengers"])

    assert frame.to_dict(orient="records") == [
        {"Date": "6/3/2026", "Passengers": "2,400,663"},
        {"Date": "6/2/2026", "Passengers": "2,218,178"},
    ]
