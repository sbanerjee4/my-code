import java.util.*;
import java.io.*;
import java.lang.*;

public class robo
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("robo.in"));
      int n = in.nextInt();
      Stack<Integer> stack = new Stack<>();
      int count = 1;
      
      for(int i = 0; i < n; i++)
      {
         if(in.next().equals("ADD"))
         {
            stack.push(count);
            count++;
         }
         else
            stack.pop();
      }
      
      in.close();
      
      Stack<Integer> stack2 = new Stack<>();
      while(!stack.isEmpty())
         stack2.push(stack.pop());
      
      System.out.println(stack2.size());
      while(!stack2.isEmpty())
         System.out.println(stack2.pop());
   }
}