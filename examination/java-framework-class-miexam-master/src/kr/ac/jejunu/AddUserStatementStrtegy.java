package kr.ac.jejunu;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

/**
 * Created by masinogns on 2017. 4. 21..
 */
public class AddUserStatementStrtegy implements StatementStrategy {

    Product product;
    public AddUserStatementStrtegy(Product product) {
        this.product = product;
    }

    @Override
    public PreparedStatement makeStatement(Connection connection) throws SQLException {

        PreparedStatement preparedStatement;
        preparedStatement = connection.prepareStatement("insert into product (id, title, price) VALUES (?,?,?)");
        preparedStatement.setLong(1, product.getId());
        preparedStatement.setString(2, product.getTitle());
        preparedStatement.setInt(3, product.getPrice());
        preparedStatement.executeUpdate();
        return preparedStatement;
    }
}
