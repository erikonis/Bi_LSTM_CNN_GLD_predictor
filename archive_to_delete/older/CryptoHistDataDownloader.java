import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

//FIXME: Implement crumb cookie or find a way to resolve the requests.

public class CryptoHistDataDownloader {
    public static void main(String[] args) {
        // debugRequests();

        String[] cryptos = { "BTC-USD",
                "ADA-USD",
                "LTC-USD",
                "LINK-USD",
                "BNB-USD",
                "VET-USD",
                "ETH-USD",
                "CRO-USD",
                "DOGE-USD",
                "USDT-USD",
                "XRP-USD",
                "AVAX-USD",
        };

        System.out
                .println("Available cryptocurrencies: BTC, ADA, LTC, LINK, BNB, VET, ETH, CRO, DOGE, USDT, XRP, AVAX");
        System.out.println(
                "Historical data will be downloaded as a CSV file in the same directory as this file.");

        Scanner scan = new Scanner(System.in);
        System.out.println("Please enter '1' to be able to select a cryptocurrency; '2' to download all.");
        int choice = scan.nextInt();

        switch (choice) {
            case 1:
                promptUser(scan, cryptos);
                break;
            case 2:
                downloadAll(cryptos);
                break;
            default:
                System.out.println("Wrong choice selected. Terminating.");
        }

    }

    public static void downloadAll(String[] cryptos) {
        for (String crypto : cryptos) {
            try {
                downloadCsv(crypto, LocalDate.ofEpochDay(0), LocalDate.now(), Path.of(crypto + ".csv"));
            } catch (Exception e) {
                System.err.println("Failed to fetch data for " + crypto + "!");
                System.err.println(e.getMessage());
                e.printStackTrace();
            }
        }
    }

    // returns epoch seconds at 00:00 UTC for a LocalDate
    private static long toEpochSeconds(LocalDate d) {
        return d.atStartOfDay().toEpochSecond(ZoneOffset.UTC);
    }

    /**
     * Sends API call to fetch a preselected crypto price data and saves it in the
     * outFile path.
     * 
     * @param ticker  String. The crypto to get data from. E.g., "BTC-USD".
     * @param start   LocalDate. Time of the crypto data. For the oldest possible
     *                date, pass LocalDate.ofEpochDay(0)
     * @param end     LocalDate. Time of the crypto data. For the latest:
     *                LocalDate.now()
     * @param outFile Path. The path of the file to save the results. Recommended:
     *                Path.of("output.csv")
     * @throws Exception
     */

    public static void downloadCsv(String ticker, LocalDate start, LocalDate end, Path outFile) throws Exception {
        long period1 = toEpochSeconds(start);
        // Yahoo's "period2" param is exclusive; use end+1 day to include 'end'
        long period2 = toEpochSeconds(end.plusDays(1));

        CrumbAndCookies cc = fetchCrumbAndCookies(ticker);
        String crumbParam = URLEncoder.encode(cc.crumb, StandardCharsets.UTF_8);

        String url = String.format(
                "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%d&period2=%d&interval=1d&events=history&includeAdjustedClose=true&crumb=%s",
                ticker, period1, period2, crumbParam);

        // temp file + atomic move pattern
        Path tmp = outFile.resolveSibling(outFile.getFileName().toString() + ".part");
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("User-Agent", "Java HttpClient")
                .header("Accept", "text/csv,text/plain,*/*")
                .header("Referer", "https://finance.yahoo.com/quote/" + ticker + "/history")
                .header("Cookie", cc.cookieHeader)
                .GET()
                .build();

        HttpResponse<Path> resp = client.send(req, HttpResponse.BodyHandlers.ofFile(tmp));
        int code = resp.statusCode();
        if (code >= 200 && code < 300) {
            try {
                java.nio.file.Files.move(tmp, outFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException ex) {
                java.nio.file.Files.move(tmp, outFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            return;
        } else {
            java.nio.file.Files.deleteIfExists(tmp);
            throw new RuntimeException("Failed to download CSV, status: " + code);
        }
    }

    public static void promptUser(Scanner scan, String[] cryptos) {
        while (true) {
            System.out.print("Please select the cryptocurrency: (enter 'q' to quit) or ('2' to download all)\n> ");
            String choice = scan.next().trim();

            if ("q".equalsIgnoreCase(choice)) {
                // exit the prompt loop
                return;
            }

            if ("2".equals(choice)) {
                // download all using the same list as in main
                downloadAll(cryptos);
                continue;
            }

            String crypto;
            switch (choice.toUpperCase()) {
                case "BTC":
                    crypto = "BTC-USD";
                    break;
                case "ADA":
                    crypto = "ADA-USD";
                    break;
                case "LTC":
                    crypto = "LTC-USD";
                    break;
                case "LINK":
                    crypto = "LINK-USD";
                    break;
                case "BNB":
                    crypto = "BNB-USD";
                    break;
                case "VET":
                    crypto = "VET-USD";
                    break;
                case "ETH":
                    crypto = "ETH-USD";
                    break;
                case "CRO":
                    crypto = "CRO-USD";
                    break;
                case "DOGE":
                    crypto = "DOGE-USD";
                    break;
                case "USDT":
                    crypto = "USDT-USD";
                    break;
                case "XRP":
                    crypto = "XRP-USD";
                    break;
                case "AVAX":
                    crypto = "AVAX-USD";
                    break;
                default:
                    System.out.println(
                            "Cryptocurrency not found. Please make sure that you select a cryptocurrency from the list.");
                    continue;
            }

            try {
                downloadCsv(crypto, LocalDate.ofEpochDay(0), LocalDate.now(), Path.of(crypto + ".csv"));
                System.err.println("Downloaded: " + crypto + " -> " + crypto + ".csv");
            } catch (Exception e) {
                System.err.println("Failed to fetch data for " + crypto + "!");
                e.printStackTrace();
            }
        }
    }

    public static void debugRequests() {
        String ticker = "BTC-USD";
        long period2 = toEpochSeconds(LocalDate.now());
        long period1 = toEpochSeconds(LocalDate.now().minusDays(1));

        String url = String.format(
                "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%d&period2=%d&interval=1d&events=history&includeAdjustedClose=true",
                ticker, period1, period2);

        // debug: print status, headers and first 2KB of body
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("User-Agent", "Java HttpClient")
                .header("Accept", "text/csv,text/plain,*/*")
                .header("Referer", "https://finance.yahoo.com/quote/" + ticker + "/history")
                .GET()
                .build();
        try {
            HttpResponse<String> r = client.send(req, HttpResponse.BodyHandlers.ofString());
            System.out.println("status=" + r.statusCode());
            System.out.println("headers=" + r.headers().map());
            System.out.println(r.body().substring(0, Math.min(r.body().length(), 2048)));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static class CrumbAndCookies {
        final String crumb;
        final String cookieHeader;

        CrumbAndCookies(String crumb, String cookieHeader) {
            this.crumb = crumb;
            this.cookieHeader = cookieHeader;
        }
    }

    private static CrumbAndCookies fetchCrumbAndCookies(String ticker) throws Exception {
        String historyUrl = "https://finance.yahoo.com/quote/" + ticker + "/history";
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(historyUrl))
                .header("User-Agent", "Java HttpClient")
                .GET().build();

        HttpResponse<String> r = client.send(req, HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200)
            throw new RuntimeException("Failed to fetch history page: " + r.statusCode());

        // Build Cookie header from Set-Cookie entries (take name=value before first
        // ';')
        List<String> setCookies = r.headers().allValues("set-cookie");
        StringBuilder cookieHeader = new StringBuilder();
        for (String sc : setCookies) {
            String nv = sc.split(";", 2)[0];
            if (cookieHeader.length() > 0)
                cookieHeader.append("; ");
            cookieHeader.append(nv);
        }

        // Extract crumb from page (CrumbStore pattern)
        Pattern p = Pattern.compile("\"CrumbStore\":\\{\"crumb\":\"(?<crumb>[^\"\\\\]+)\"\\}");
        Matcher m = p.matcher(r.body());
        if (!m.find()) {
            // sometimes crumb contains escaped unicode; try a looser pattern
            p = Pattern.compile("CrumbStore\":\\{\"crumb\":\"(.*?)\"\\}");
            m = p.matcher(r.body());
            if (!m.find())
                throw new RuntimeException("Crumb not found on history page");
        }
        String crumb = m.group("crumb");
        // crumb may contain escaped sequences like \\u002F — unescape common ones
        crumb = crumb.replace("\\u002F", "/").replace("\\\\", "\\");
        return new CrumbAndCookies(crumb, cookieHeader.toString());
    }
}