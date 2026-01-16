package utils;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;

public class PrefixFormatter extends Formatter {
    private final String prefix;
    private final SimpleDateFormat dateFormat = new SimpleDateFormat("MM/dd/yy HH:mm:ss");

    public PrefixFormatter(String prefix) {
        this.prefix = prefix;
    }

    @Override
    public String format(LogRecord record) {
        // 1. Format the timestamp
        String timestamp = dateFormat.format(new Date(record.getMillis()));

        // 2. Format the level to be consistent (optional: add padding for alignment)
        String level = String.format("%-7s", record.getLevel().getName());

        // 3. Construct the final string
        // Format: [MM/dd/yy HH:mm:ss] LEVEL Message
        return String.format("[%s] %s %s%n",
                timestamp,
                level,
                record.getMessage());
    }
}